import os
import shutil
import re
import json
import yaml
from huggingface_hub import Repository, create_repo
from pathlib import Path
from PIL import Image, ImageDraw
import io
import torch
from torchvision import transforms
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
import albumentations as A
import logging
#creates a logger for the current module
logger = logging.getLogger(__name__)

import numpy as np
from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers.image_transforms import center_to_corners_format
from DeepDataMiningLearning.detection.plotutils import draw2pil, pixel_values2img, draw_objectdetection_predboxes, draw_objectdetection_results

def convert_bbox_yolo_to_pascal(boxes, image_size):
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes

@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


@torch.no_grad()
def compute_mapmetrics(evaluation_results, image_processor, threshold=0.0, id2label=None):
    """
    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """

    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # Collect targets in the required format for metric computation
    for batch in targets:
        # collect image sizes, we will need them for predictions post processing
        batch_image_sizes = torch.tensor(np.array([x["orig_size"] for x in batch]))
        image_sizes.append(batch_image_sizes)
        # collect targets in the required format for metric computation
        # boxes were converted to YOLO format needed for model training
        # here we will convert them to Pascal VOC format (x_min, y_min, x_max, y_max)
        for image_target in batch:
            boxes = torch.tensor(image_target["boxes"])
            boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

    return metrics


#from https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py
PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


def load_hfcheckpoint(checkpoint_dir, overwrite_output_dir=False):
    last_checkpoint = None
    if checkpoint_dir is not None and os.path.isdir(checkpoint_dir) and not overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(checkpoint_dir)
        if last_checkpoint is None and len(os.listdir(checkpoint_dir)) > 0:
            raise ValueError(
                f"Output directory ({checkpoint_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint

def freeze_model(model, partname='classifier'):
    for name,p in model.named_parameters():
        if not name.startswith(partname):
            p.requires_grad = False

def calculate_params(model):
    num_params = sum([p.numel() for p in model.parameters()])
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])

    print(f"{num_params = :,} | {trainable_params = :,}")
    
def load_config(config_path):
    """Loads configuration parameters from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format in {config_path}: {e}")
        return None

def load_config_relativefolder(filename="config.yaml"):
    """
    Loads configuration parameters from a YAML file.

    Args:
        filename (str): The name or path of the YAML file to load.

    Returns:
        dict: A dictionary containing the configuration parameters, or None if an error occurs.
    """
    try:
        # Check if the filename is a path
        if os.path.isabs(filename) or os.path.exists(filename):
            config_path = filename
        else:
            # Get the directory of the current script and combine with the filename
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if not filename.lower().endswith(".yaml"):
                filename += ".yaml"
            config_path = os.path.join(script_dir, "configs", filename)

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format in {config_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def pushtohub(hub_model_id, output_dir, hub_token):
    # Retrieve of infer repo_name
    repo_name = hub_model_id
    if repo_name is None:
        repo_name = Path(output_dir).absolute().name
    # Create repo and retrieve repo_id
    repo_id = create_repo(repo_name, exist_ok=True, token=hub_token).repo_id
    # Clone repo locally
    repo = Repository(output_dir, clone_from=repo_id, token=hub_token)

    with open(os.path.join(output_dir, ".gitignore"), "w+") as gitignore:
        if "step_*" not in gitignore:
            gitignore.write("step_*\n")
        if "epoch_*" not in gitignore:
            gitignore.write("epoch_*\n")

def readimg2PILRGB(example_batch_imgs):
    processed_images = []
    for image_data in example_batch_imgs:
        if isinstance(image_data, Image.Image):
            # Image is already a PIL Image object
            image = image_data
        else:
            try:
                # Attempt to open image from bytes
                image = Image.open(io.BytesIO(image_data['bytes']))
            except Exception as e:
                print(f"Error opening image: {e}")
                #Handle the error. Either skip, or provide a default image.
                #Example, skipping the image:
                continue
                #Or return a default image, for example a blank white image:
                #image = Image.new('RGB', (1, 1), color = 'white')
        processed_images.append(image.convert("RGB"))
    return processed_images

def get_data_transform(image_processor, has_boundingbox=False, image_maxsize=None):

    if has_boundingbox == False:
        if image_maxsize:
            size = (image_maxsize, image_maxsize)
        elif "shortest_edge" in image_processor.size:
            size = image_processor.size["shortest_edge"]
        elif "height" in image_processor.size:
            size = (image_processor.size["height"], image_processor.size["width"]) #(224, 224)
        else:
            size = (224, 224)
        
        image_mean = [0.485, 0.456, 0.406 ]
        image_std = [0.229, 0.224, 0.225]
        normalize = (
                Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
                if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std")
                else Normalize(mean=image_mean, std=image_std) #Lambda(lambda x: x)
            )
        train_transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )
        val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )
    else:
        train_transforms = A.Compose(
            [
                #A.Perspective(p=0.1),
                A.Resize(640,640),
		A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.1),
            ],
            bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
        )
        val_transforms = A.Compose(
            [A.NoOp()],
            bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
        )
    
    return train_transforms, val_transforms

# if not isinstance(size, tuple):
#             size = (size, size)
#         train_transforms = albumentations.Compose(
#             [
#                 albumentations.Resize(height=size[0], width=size[1]), #(480, 480),
#                 albumentations.HorizontalFlip(p=1.0),
#                 albumentations.RandomBrightnessContrast(p=1.0),
#             ],
#             bbox_params=albumentations.BboxParams(format=format, min_area=1024, min_visibility=0.1, label_fields=["category"]),
#         )
#         val_transforms = albumentations.Compose(
#             [
#                 albumentations.Resize(height=size[0], width=size[1]), #(480, 480),
#             ],
#             bbox_params=albumentations.BboxParams(format=format, min_area=1024, min_visibility=0.1, label_fields=["category"]),
#         )

#For pixel_values, the input shape for the model should be (batch, channels, height, width) and \
    # for labels, the shape should be (batch,)
def get_collate_fn(task, image_processor, label_column_name=None, require_padding=False):
    def get_labelslist(examples):
        labels_list = []
        for example in examples:
            label = example[label_column_name]
            if not isinstance(label, int):
                #print(f"Error: Label '{label}' in example is not an integer.")
                #return None  # Return None if a non-integer label is found
                labels_list.append(int(label))
            else:
                labels_list.append(label)
        return labels_list
    
    if task == "image-classification":
        # DataLoaders creation:
        #used to batch examples together. Each batch consists of 2 keys, namely pixel_values and labels.
        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            labels_list=get_labelslist(examples)
            labels = torch.tensor(labels_list)
            #labels = torch.tensor([example[label_column_name] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}
        #similar to 
        #collate_fn = DefaultDataCollator()
        return collate_fn
    elif task == "object-detection":
        # batch images together. Pad images (which are now pixel_values) to the largest image in a batch, 
        #and create a corresponding pixel_mask to indicate which pixels are real (1) and which are padding (0).
        def object_detection_collate_fn_padding(batch):#batch is a list input
            pixel_values = [item["pixel_values"] for item in batch] #list of [3,800,800]
            #'DetrImageProcessor' object has no attribute 'pad_and_create_pixel_mask'
            #encoding = image_processor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
            encoding = image_processor.pad(pixel_values, return_tensors="pt")
            labels = [item["labels"] for item in batch]#pixel_values [8,3,800,800]
            batch = {} #keep the other keys in the batch
            batch["pixel_values"] = encoding["pixel_values"]
            if "pixel_mask" in encoding:
                batch["pixel_mask"] = encoding["pixel_mask"] #[8,800,800]
            batch["labels"] = labels #8 dict items
            return batch
        def object_detection_collate_fn(batch):
            data = {}
            data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
            data["labels"] = [x["labels"] for x in batch]
            if "pixel_mask" in batch[0]:
                data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
            return data
        if require_padding:
            return object_detection_collate_fn_padding
        else:
            return object_detection_collate_fn


def dataset_objectdetection_select(dataset, data_index, labels, format='coco', image_column_name='image', label_column_name='objects', output_folder="output/"):
    image = dataset[data_index][image_column_name]#PIL image in RGB mode 
    annotations = dataset[data_index][label_column_name] 
    #['id'] ['area'] ['bbox'](4,4)list ['category']
    #in coco format https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#coco
    #bounding box in [x_min, y_min, width, height]
    filepath = os.path.join(output_folder, "dataset_objectdetection_select.png")
    image_annoted = draw2pil(image, annotations['bbox'], labels, format, filepath)
    print(f"Test image id:{data_index} saved in {filepath}")

    transform = A.Compose([
        A.Resize(480, 480),
        A.HorizontalFlip(p=1.0),
        A.RandomBrightnessContrast(p=1.0),
    ], bbox_params=A.BboxParams(format=format,  label_fields=['category']))
    image_np = np.array(image) #HWC
    out = transform(
        image=image_np,
        bboxes=annotations['bbox'],
        category=annotations['category'],
    )
    print(out.keys()) #['image', 'bboxes', 'category']

    image = torch.tensor(out['image']).permute(2, 0, 1) #HWC->CHW
    boxes_xywh = torch.stack([torch.tensor(x) for x in out['bboxes']])
    #pil_image=Image.fromarray(np.uint8(out['image']))
    filepath = os.path.join(output_folder, "dataset_objectdetection_transform.png")
    image_annoted = draw2pil(image, boxes_xywh, labels, format, filepath)
    print(f"Test image id:{data_index} transformed saved in {filepath}")
    # boxes_xyxy = box_convert(boxes_xywh, 'xywh', 'xyxy')
    # labels = [categories.int2str(x) for x in out['category']]
    # to_pil_image(
    #     draw_bounding_boxes(
    #         image,
    #         boxes_xyxy,
    #         colors='red',
    #         labels=labels
    #     )
    # )

       # dataset_objectdetection_select(split_datasets["train"], data_index=0, id2label=id2label, categories=categories, format=format, image_column_name=image_column_name, label_column_name=label_column_name, output_folder="output/")

def checkimage_aftertransform(dataset, train_dataset, labels=None, data_index=0, format='coco', image_column_name='image', label_column_name='objects', output_folder="output/"):
    image = dataset["train"][data_index][image_column_name]#PIL image in RGB mode 
    annotations = dataset["train"][data_index][label_column_name] 
    #['id'] ['area'] ['bbox'](4,4)list ['category']
    #in coco format https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#coco
    #bounding box in unnormalized [x_min, y_min, width, height]
    filepath = os.path.join(output_folder, "dataset_objectdetection_sample.png")
    image_annoted = draw2pil(image, annotations['bbox'], labels, format, filepath)
    print(f"Test image id:{data_index} saved in {filepath}")

    #train_dataset = dataset["train"].map(preprocess_train)
    #eval_dataset = dataset[valkey].map(preprocess_val)
    oneexample = train_dataset[data_index] #perform preprocessing
    print(oneexample.keys()) 
    #'pixel_values'[3, 800, 800], 'pixel_mask'[800,800], 'labels'dict of 'boxes'[2,4] (center_x, center_y, width, height) normalized

    filepath = os.path.join(output_folder, "dataset_objectdetection_transform.png")
    anno = oneexample['labels']
    #image is PIL, anno has keys of size (480, 480), image_id, class_labels, boxes (normalized) in tensor
    #boxes (normalized) in tensor (center_x, center_y, width, height) normalized
    image_annoted = draw2pil(oneexample['pixel_values'], anno['boxes'], labels, bbox_format=None, filepath=filepath)
    print(f"Test image id:{data_index} saved in {filepath}")

from DeepDataMiningLearning.detection.dataset_hf import check_boxsize
def preprocessimage_objectdetection_old(examples, image_transforms, image_column_name, label_column_name):
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples[image_column_name], examples[label_column_name]): #label_column_name="objects"
        #image = np.array(image.convert("RGB"))[:, :, ::-1] #(720, 1280, 3)HWC,, BGR?
        image = np.array(image.convert("RGB"))
        height, width, channel = image.shape
        # if format != 'coco':
        #     bbox_new = convert_bbbox2coco(objects["bbox"], source_format=format) #[x_min, y_min, width, height]
        # else:
        #     bbox_new = objects["bbox"]
        
        bbox_new = objects["bbox"]
        newbbox, errbox = check_boxsize(bbox_new, height=height, width=width, format=format)
        if errbox:
            print(bbox_new)
        out = image_transforms(image=image, bboxes=newbbox, category=objects["category"])#already consider format bbox size changed
        
        area.append(objects["area"])
        images.append(out[image_column_name]) #(480, 480, 3)
        bboxes.append(out["bboxes"])#resized [x_min, y_min, width, height]
        categories.append(out["category"])#category become integer list [4]
    return images, bboxes, area, categories

def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
        areas (List[float]): list of corresponding areas to provided bounding boxes
        bboxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
            ([center_x, center_y, width, height] in absolute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }

def preprocessimage_objectdetection(examples, image_transforms, image_column_name, label_column_name):
    """Apply augmentations and format annotations in COCO format for object detection task"""
    images = []
    annotations = []
    # print(type(examples["image_id"]), examples["image_id"])
    # print(type(examples["image"]), examples["image"])
    # print(type(examples["objects"]), examples["objects"])

    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB")) #HWC format
        height, width, channel = image.shape

        bbox_new = objects["bbox"]
        newbbox, errbox = check_boxsize(bbox_new, height=height, width=width, format="coco")
        if errbox:
            print("Detected errbox:",bbox_new)

        # apply augmentations
        output = image_transforms(image=image, bboxes=newbbox, category=objects["category"])
        images.append(output["image"])

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"]
        )
        annotations.append(formatted_annotations)
    return images, annotations
