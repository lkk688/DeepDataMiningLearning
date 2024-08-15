
import argparse
import json
import logging
import math
import os
from pathlib import Path
import datetime
import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, DatasetDict
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
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
import requests
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import classification_report
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification, \
    AutoModelForDepthEstimation, AutoModelForObjectDetection, SchedulerType, get_scheduler
from transformers import DefaultDataCollator, Trainer, TrainingArguments
from time import perf_counter
from DeepDataMiningLearning.visionutil import get_device, saveargs2file, load_ImageNetlabels, read_image
import requests
import cv2
import albumentations#pip install albumentations
from DeepDataMiningLearning.detection.dataset_hf import HFCOCODataset, check_boxsize
from DeepDataMiningLearning.detection.plotutils import draw2pil, pixel_values2img, draw_objectdetection_predboxes, draw_objectdetection_results

logger = get_logger(__name__)

#The PYTORCH_USE_CUDA_DSA environment variable is used to enable the use of the CUDA Direct Storage API (DSA) in PyTorch. DSA is a new API that allows PyTorch to directly access data on the GPU without having to copy it to the CPU first.
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"

class myEvaluator:
    def __init__(self, task, useHFevaluator=False, dualevaluator=False, processor=None, coco=None, mycache_dir=None):
        print("useHFevaluator:", useHFevaluator)
        print("dualevaluator:", dualevaluator)
        self.useHFevaluator = useHFevaluator
        self.dualevaluator = dualevaluator
        self.task = task
        self.preds = []
        self.refs = []
        #self.labels = labels
        self.processor = processor
        self.HFmetric = None
        if self.task == "image-classification":
            self.metricname = "accuracy" #"mse" "wer"
        elif self.task == "object-detection":
            self.metricname = "coco"
            #prepare
        else:
            self.metricname = "accuracy"
        self.LOmetric = None
        if self.useHFevaluator and self.task=="object-detection":
            self.HFmetric = evaluate.load("ybelkada/cocoevaluate", coco=coco) #test_ds_coco_format.coco)
        elif self.useHFevaluator:
            # Load the accuracy metric from the datasets package
            self.HFmetric = evaluate.load(self.metricname, cache_dir=mycache_dir) #evaluate.load("mse")
            

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with
    # `predictions` and `label_ids` fields) and has to return a dictionary string to float.
    #eval_pred is EvalPrediction type
    def compute_metrics(self, eval_pred): #: EvalPrediction):
        #preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions #(1000, 593, 46)
        preds, labels = eval_pred
        if self.metricname == "accuracy":
            """Computes accuracy on a batch of predictions"""
            preds = np.argmax(preds, axis=1)
            #return self.HFmetric.compute(predictions=predictions, references=labels)
        elif self.metricname == "mse":
            preds = np.squeeze(preds)
            #return self.HFmetric.compute(predictions=preds, references=label_ids)
        return self.compute(predictions=preds, references=labels)

    def mycompute(self, predictions=None, references=None):
        predictions = np.array(predictions)
        references = np.array(references)
        if self.metricname == "accuracy":
            eval_result = (predictions == references).astype(np.float32).mean().item()
            # if self.labels:
            #     print("Classification report", classification_report(references, predictions, target_names=self.labels))
        elif self.metricname == "mse": #mse
            eval_result = ((predictions - references) ** 2).mean().item()
        results = {self.metricname: eval_result}
        return results
    
    def compute(self, predictions=None, references=None):
        results = {}
        if predictions is not None and references is not None:
            if self.useHFevaluator:
                results = self.HFmetric.compute(predictions=predictions, references=references)
            else: 
                results = self.mycompute(predictions=predictions, references=references)
            #print("HF evaluator:", results)
            if not isinstance(results, dict):
                #output is float, convert to dict
                results = {self.metricname: results}
        else: #evaluate the whole dataset
            if self.useHFevaluator:
                results = self.HFmetric.compute()
                print("HF evaluator result1:", results)#iou_bbox 
                #results2 = self.HFmetric.compute(predictions=self.preds, references=self.refs) #the same results
                #print("HF evaluator result2:", results2)
                if not isinstance(results, dict):
                    #wer output is float, convert to dict
                    results = {self.metricname: results}
            else:
                results = self.mycompute(predictions=self.preds, references=self.refs)
            self.preds.clear()
            self.refs.clear()
        if self.task == "object-detection":
            results = results['iou_bbox']
        return results
    
    def add_batch(self, predictions, references):
        if self.useHFevaluator == True:
            if self.task=="object-detection":
                self.HFmetric.add(prediction=predictions, reference=references)
            else:
                self.HFmetric.add_batch(predictions=predictions, references=references)
        else:
            #self.preds.append(predictions)
            self.refs.extend(references)
            self.preds.extend(predictions)
        #references: list of list
        # for ref in references:
        #     self.refs.append(ref[0])
        #print(len(self.refs))

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

valkey='test' #"validation"
#data_name list: food101, 
def load_visiondataset(data_name=None, split="train", train_dir=None, validation_dir=None, task="image-classification", format='coco', max_train_samples = 2000, train_val_split=0.15, \
                       image_column_name='image', label_column_name='labels', mycache_dir=None):
    if data_name is not None:
        if max_train_samples and max_train_samples>0 and split is not None:
            data_split=f"{split}[:{max_train_samples}]" #"train+validation"
        elif split is not None:
            data_split=f"{split}"
        else:
            data_split=None
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(data_name,split=data_split, cache_dir=mycache_dir, verification_mode='no_checks')#, trust_remote_code=True) #ignore_verifications=True
        # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.html.)
    else:
        data_files = {}
        if train_dir is not None:
            data_files["train"] = os.path.join(train_dir, "**")
        if validation_dir is not None:
            data_files[valkey] = os.path.join(validation_dir, "**")
        raw_datasets = load_dataset(
            "imagefolder",
            data_files=data_files,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder.
    #splits=raw_datasets.split
    # print(raw_datasets.columns)
    # If we don't have a validation split, split off a percentage of train as validation.
    #args.train_val_split = None if "validation" in dataset.keys() else args.train_val_split
    split_datasets = DatasetDict()
    if isinstance(raw_datasets.column_names, dict):#'train' 'test' key
        print("All keys in raw datasets:", raw_datasets['train']) #
        if valkey not in raw_datasets.keys():
            split = raw_datasets["train"].train_test_split(test_size=train_val_split, seed=20)
            split_datasets["train"] = split["train"]
            split_datasets[valkey] = split["test"]
        else:
            split_datasets = raw_datasets
    else: #no train/test split
        split_datasets["train"] = raw_datasets
        split_datasets = split_datasets["train"].train_test_split(test_size=train_val_split, seed=20) #get splits
        if valkey!="test":
            # rename the "test" key to "validation" 
            split_datasets[valkey] = split_datasets.pop("test")

    #limit the dataset size
    if max_train_samples>0 and len(split_datasets['train'])>max_train_samples:
        split_datasets['train'] = split_datasets['train'].select([i for i in list(range(max_train_samples))])
        Val_SAMPLES = int(max_train_samples*train_val_split)
        split_datasets[valkey] = split_datasets[valkey].select([i for i in list(range(Val_SAMPLES))])
    
    dataset_column_names = split_datasets["train"].column_names if "train" in split_datasets else split_datasets[valkey].column_names
    #'image': PIL image object, 'labels': int, 'image_file_path': path
    
    if task == "object-detection":
        image_column_name = "image"
        label_column_name = "objects" #for object detection
        # remove_idx = [590, 821, 822, 875, 876, 878, 879]
        # keep = [i for i in range(len(split_datasets["train"])) if i not in remove_idx]
        # split_datasets["train"] = split_datasets["train"].select(keep)
    elif task == "image-classification":
        #some datset the labels name is different
        if data_name == "food101": #https://huggingface.co/datasets/food101
            image_column_name = "image"
            label_column_name = "label"
        elif 'cats_vs_dogs' in data_name:
            image_column_name = "image"
            label_column_name = "labels"

    if image_column_name not in dataset_column_names:
        raise ValueError(
            f"--image_column_name {image_column_name} not found in dataset '{data_name}'. "
            "Make sure to set `--image_column_name` to the correct audio column - one of "
            f"{', '.join(dataset_column_names)}."
        )
    if label_column_name not in dataset_column_names:
        raise ValueError(
            f"--label_column_name {label_column_name} not found in dataset '{data_name}'. "
            "Make sure to set `--label_column_name` to the correct text column - one of "
            f"{', '.join(dataset_column_names)}."
        )
    
    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    #By default the ClassLabel fields are encoded into integers
    if task == "image-classification":
        classlabel = split_datasets["train"].features[label_column_name] #ClassLabel(num_classes=x, names=[''], id=None)
        labels = classlabel.names
        label2id = {label: str(i) for i, label in enumerate(labels)}
        id2label = {str(i): label for i, label in enumerate(labels)}
            #add testing code to fetch one sample data from dataset and print the shape of data
        # {
        #     "image": "train/cat/00000.png",
        #     "label": 0
        # }
        #print(split_datasets["train"][0])
        print("Classification dataset 0:", split_datasets["train"][0][label_column_name])
    elif task=="object-detection":
        classlabel = split_datasets["train"].features[label_column_name]#Sequence class, feature['category','area','box','category'], id
        categories = classlabel.feature["category"]#Classlabel class
        labels = categories.names #list of str names
        id2label = {index: x for index, x in enumerate(labels, start=0)}
        label2id = {v: k for k, v in id2label.items()}
        dataset_objectdetection_select(split_datasets["train"], data_index=0, id2label=id2label, categories=categories, format=format, \
                                       image_column_name=image_column_name, label_column_name=label_column_name, output_folder="output/")


    #The Datasets library is made for processing data very easily. We can write custom functions, 
    #which can then be applied on an entire dataset (either using .map() or .set_transform()).
    return split_datasets, labels, id2label, image_column_name, label_column_name


def dataset_objectdetection_select(dataset, data_index, id2label, categories, format='coco', image_column_name='image', label_column_name='objects', output_folder="output/"):
    image = dataset[data_index][image_column_name]#PIL image in RGB mode 
    annotations = dataset[data_index][label_column_name] 
    #['id'] ['area'] ['bbox'](4,4)list ['category']
    #in coco format https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#coco
    #bounding box in [x_min, y_min, width, height]
    filepath = os.path.join(output_folder, "dataset_objectdetection_select.png")
    image_annoted = draw2pil(image, annotations['bbox'], annotations['category'], categories, format, filepath)
    print(f"Test image id:{data_index} saved in {filepath}")

    transform = albumentations.Compose([
        albumentations.Resize(480, 480),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.RandomBrightnessContrast(p=1.0),
    ], bbox_params=albumentations.BboxParams(format=format,  label_fields=['category']))
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
    image_annoted = draw2pil(image, boxes_xywh, out['category'], categories, format, filepath)
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

def formatted_anns(image_id, category, area, bbox):#category/area/bbox list input
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations #list of dicts


def dataset_preprocessing(image_processor, task, size, format='coco', image_column_name='image', label_column_name='labels'):
    image_mean = [0.485, 0.456, 0.406 ]
    image_std = [0.229, 0.224, 0.225]
    normalize = (
            Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
            if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std")
            else Normalize(mean=image_mean, std=image_std) #Lambda(lambda x: x)
        )
    if task =="image-classification":
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
        def preprocess_train(example_batch):
            """Apply _train_transforms across a batch."""
            #PIL image to RGB
            example_batch["pixel_values"] = [
                train_transforms(image.convert("RGB")) for image in example_batch[image_column_name]
            ]
            del example_batch[image_column_name]
            return example_batch

        def preprocess_val(example_batch):
            """Apply _val_transforms across a batch."""
            example_batch["pixel_values"] = [
                val_transforms(image.convert("RGB")) for image in example_batch[image_column_name]
            ]
            del example_batch[image_column_name]
            return example_batch
    
    elif task =="object-detection":
        if not isinstance(size, tuple):
            size = (size, size)
        train_transforms = albumentations.Compose(
            [
                albumentations.Resize(height=size[0], width=size[1]), #(480, 480),
                albumentations.HorizontalFlip(p=1.0),
                albumentations.RandomBrightnessContrast(p=1.0),
            ],
            bbox_params=albumentations.BboxParams(format=format, min_area=1024, min_visibility=0.1, label_fields=["category"]),
        )
        val_transforms = albumentations.Compose(
            [
                albumentations.Resize(height=size[0], width=size[1]), #(480, 480),
            ],
            bbox_params=albumentations.BboxParams(format=format, min_area=1024, min_visibility=0.1, label_fields=["category"]),
        )
        #The image_processor expects the annotations to be in the following format: {'image_id': int, 'annotations': List[Dict]}, 
        #where each dictionary is a COCO object annotation.
        # transforming a batch
        def preprocess_train(examples):#can handle batch
            image_ids = examples["image_id"]
            images, bboxes, area, categories = [], [], [], []
            for image, objects in zip(examples[image_column_name], examples[label_column_name]): #label_column_name="objects"
                image = np.array(image.convert("RGB"))[:, :, ::-1] #(720, 1280, 3)HWC
                height, width, channel = image.shape
                # if format != 'coco':
                #     bbox_new = convert_bbbox2coco(objects["bbox"], source_format=format) #[x_min, y_min, width, height]
                # else:
                #     bbox_new = objects["bbox"]
                
                bbox_new = objects["bbox"]
                newbbox, errbox = check_boxsize(bbox_new, height=height, width=width, format=format)
                if errbox:
                    print(bbox_new)
                out = train_transforms(image=image, bboxes=newbbox, category=objects["category"])#already consider format bbox size changed
                
                area.append(objects["area"])
                images.append(out[image_column_name]) #(480, 480, 3)
                bboxes.append(out["bboxes"])#resized [x_min, y_min, width, height]
                categories.append(out["category"])#category become integer list [4]

            targets = [
                {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
                for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
            ]#list of dict 'image_id'=756, 'annotations': list of dicts with 'area' 'bbox' 'category_id'

            #https://huggingface.co/docs/transformers/main/en/model_doc/detr#transformers.DetrImageProcessor
            return image_processor(images=images, annotations=targets, return_tensors="pt")
            #do_convert_annotations: Converts the bounding boxes from the format (top_left_x, top_left_y, width, height) to (center_x, center_y, width, height) and in relative coordinates.
            #input_data_format: "channels_first" CHW, "channels_last" HWC
            #If unset, the channel dimension format is inferred from the input image.
        
        def preprocess_val(examples):#can handle batch
            image_ids = examples["image_id"]
            images, bboxes, area, categories = [], [], [], []
            for image, objects in zip(examples[image_column_name], examples[label_column_name]): #label_column_name="objects"
                image = np.array(image.convert("RGB"))[:, :, ::-1] #(720, 1280, 3)HWC
                out = val_transforms(image=image, bboxes=objects["bbox"], category=objects["category"])#bbox size changed

                area.append(objects["area"])
                images.append(out[image_column_name]) #(480, 480, 3)
                bboxes.append(out["bboxes"])#resized [x_min, y_min, width, height]
                categories.append(out["category"])#category become integer list [4]

            targets = [
                {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
                for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
            ]#list of dict 'image_id'=756, 'annotations': list of dicts with 'area' 'bbox' 'category_id'
            #https://huggingface.co/docs/transformers/main/en/model_doc/detr#transformers.DetrImageProcessor
            return image_processor(images=images, annotations=targets, return_tensors="pt")
        
    return preprocess_train, preprocess_val


def get_collate_fn(task, image_processor, label_column_name=None):
    if task == "image-classification":
        # DataLoaders creation:
        #used to batch examples together. Each batch consists of 2 keys, namely pixel_values and labels.
        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            labels = torch.tensor([example[label_column_name] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}
        #similar to 
        #collate_fn = DefaultDataCollator()
        return collate_fn
    elif task == "object-detection":
        # batch images together. Pad images (which are now pixel_values) to the largest image in a batch, 
        #and create a corresponding pixel_mask to indicate which pixels are real (1) and which are padding (0).
        def object_detection_collate_fn(batch):#batch is a list input
            pixel_values = [item["pixel_values"] for item in batch] #list of [3,800,800]
            #'DetrImageProcessor' object has no attribute 'pad_and_create_pixel_mask'
            #encoding = image_processor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
            encoding = image_processor.pad(pixel_values, return_tensors="pt")
            labels = [item["labels"] for item in batch]#pixel_values [8,3,800,800]
            batch = {} #keep the other keys in the batch
            batch["pixel_values"] = encoding["pixel_values"]
            batch["pixel_mask"] = encoding["pixel_mask"] #[8,800,800]
            batch["labels"] = labels #8 dict items
            return batch
        return object_detection_collate_fn

#tasks: "depth-estimation", "image-classification", "object-detection"
def load_visionmodel(model_name_or_path, task="image-classification", load_only=True, labels=None, mycache_dir=None, trust_remote_code=True):
    if load_only:#only load the model
        ignore_mismatched_sizes = False
        config = None
    elif labels is not None: #Create a new model
        ignore_mismatched_sizes = True
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label
        #test convert the label id to a label name:
        #print(id2label[str(7)])
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=len(labels),
            i2label=id2label,
            label2id=label2id,
            cache_dir=mycache_dir,  
            finetuning_task=task, #"image-classification",
            trust_remote_code=trust_remote_code,
        )

    image_processor = AutoImageProcessor.from_pretrained(
        model_name_or_path,
        cache_dir=mycache_dir,
        trust_remote_code=trust_remote_code,
    )
    if task == "image-classification":
        model = AutoModelForImageClassification.from_pretrained(
            model_name_or_path,
            config=config,
            cache_dir=mycache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            trust_remote_code=trust_remote_code,
        )
    elif task == "depth-estimation":
        model = AutoModelForDepthEstimation.from_pretrained(
            model_name_or_path, 
            config=config,
            cache_dir=mycache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            trust_remote_code=trust_remote_code,
        )
    elif task == "object-detection":
        model = AutoModelForObjectDetection.from_pretrained(
            model_name_or_path, 
            config=config,
            cache_dir=mycache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            trust_remote_code=trust_remote_code,
        )
    #model.config.id2label
    return model, image_processor


def custom_train(args, model, image_processor, train_dataloader, eval_dataloader, metriceval, device, accelerator=None, do_evaluate=False):
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("image_classification_no_trainer", experiment_config)

    # Get the metric function
    #metric = evaluate.load("accuracy") #replaced with metriceval

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                if args.task == "object-detection":
                    # pixel_values = batch["pixel_values"].to(device)
                    # pixel_mask = batch["pixel_mask"].to(device)
                    # labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
                    pixel_values = batch["pixel_values"]
                    pixel_mask = batch["pixel_mask"]
                    labels = [{k: v for k, v in t.items()} for t in batch["labels"]]
                    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                    loss_dict = outputs.loss_dict
                    #print(loss_dict)
                else:
                    outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        if do_evaluate:
            eval_metric = evaluate_dataset(model, eval_dataloader, args.task, metriceval, device, image_processor=image_processor, accelerator=accelerator)
    
            # model.eval()
            # for step, batch in enumerate(eval_dataloader):
            #     with torch.no_grad():
            #         outputs = model(**batch)
            #     predictions = outputs.logits.argmax(dim=-1)
            #     predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            #     metriceval.add_batch(
            #         predictions=predictions,
            #         references=references,
            #     )

            # eval_metric = metriceval.compute()#metric.compute()

            logger.info(f"epoch {epoch}: {eval_metric}")

        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy": eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        #saving the models
        if args.checkpointing_steps == "epoch" and epoch % args.saving_everynsteps ==0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            unwrapped_model.save_pretrained(
                output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                image_processor.save_pretrained(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            image_processor.save_pretrained(args.output_dir)
            # if args.push_to_hub:
            #     repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

            if do_evaluate:
                all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
                with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                    json.dump(all_results, f)
        #push to hub
        if args.hubname:
            unwrapped_model.push_to_hub(args.hubname)
            image_processor.push_to_hub(args.hubname)

def trainmain():
    args = parse_args()
    requests.get("https://huggingface.co", timeout=5)
    #dataset = load_dataset("lhoestq/demo1")

    trainoutput=os.path.join(args.output_dir, args.data_name+'_'+args.traintag)
    os.makedirs(trainoutput, exist_ok=True)
    args.output_dir = trainoutput
    print("Trainoutput folder:", trainoutput)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    #accelerate launch --gpu_ids 6 myscript.py
    #https://huggingface.co/docs/accelerate/en/package_reference/accelerator
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    print("Accelerator device:", accelerator.device)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", #The format of the log message: 2024/02/14 16:30:00 - INFO - main
        datefmt="%m/%d/%Y %H:%M:%S", #The format of the date and time in the log message.
        level=logging.INFO, #The minimum severity level that will be logged.
    )
    #logs the current state of the accelerator to the console
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        if os.environ.get('HF_HOME') is not None:
            mycache_dir = os.environ['HF_HOME']
        elif args.data_path:
            #mycache_dir = deviceenv_set(args.usehpc, args.data_path)
            os.environ['HF_HOME'] = args.data_path
            mycache_dir = args.data_path
        else:
            mycache_dir = '~/.cache/huggingface/'
        print("Cache dir:", mycache_dir)
        device, args.useamp = get_device(gpuid=args.gpuid, useamp=args.useamp)
        saveargs2file(args, trainoutput)
    
    #waits for all processes to finish before continuing
    accelerator.wait_for_everyone()

    #load dataset
    with accelerator.main_process_first():
        dataset, labels, id2label, args.image_column_name, args.label_column_name = load_visiondataset(data_name=args.data_name, \
                                    split=args.datasplit, train_dir=args.train_dir, validation_dir=args.validation_dir, \
                                    task=args.task, format=args.format, max_train_samples = args.max_train_samples, train_val_split=args.train_val_split, \
                                    image_column_name=args.image_column_name, label_column_name=args.label_column_name, mycache_dir=mycache_dir)

    # Load pretrained model and image processor
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    model, image_processor = load_visionmodel(args.model_name_or_path, task=args.task, load_only=True, labels=labels, mycache_dir=mycache_dir, trust_remote_code=True)
    model.config.id2label = id2label

    # Preprocessing the datasets
    # Define torchvision transforms to be applied to each image.
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"]) #(224, 224)
    
    with accelerator.main_process_first():
        preprocess_train, preprocess_val = dataset_preprocessing(image_processor=image_processor, task=args.task, size=size, format=args.format, image_column_name=args.image_column_name, label_column_name=args.label_column_name)
        #The transforms are applied on the fly when you load an element of the dataset:
        train_dataset = dataset["train"].with_transform(preprocess_train)
        #train_dataset = dataset["train"].map(preprocess_train)
        #eval_dataset = dataset[valkey].map(preprocess_val)
        oneexample = train_dataset[15]
        print(oneexample.keys()) #'pixel_values'[3, 800, 800], 'pixel_mask'[800,800], 'labels'dict of 'boxes'[2,4] (center_x, center_y, width, height) normalized
        if args.task == "image-classification":
            eval_dataset = dataset[valkey].with_transform(preprocess_val)
            coco = None
        elif args.task == "object-detection":
            coco_datafolder = os.path.join(mycache_dir, 'coco_converted', args.data_name)
            eval_dataset = HFCOCODataset(dataset[valkey], id2label, dataset_folder=coco_datafolder, coco_anno_json=None, data_type=args.datatype, format=args.format, image_processor=image_processor)
            coco = eval_dataset.coco
            eval_dataset.test_cocodataset(10)
            onehfcoco = next(iter(eval_dataset)) #'pixel_values'[3, 800, 1066] 'labels'
            #print(onehfcoco)


    collate_fn = get_collate_fn(args.task, image_processor, args.label_column_name)

    metriceval = myEvaluator(task=args.task, useHFevaluator=True, dualevaluator=False, \
                            processor=image_processor, coco=coco, mycache_dir=mycache_dir)

    
    # using now() to get current time
    starting_time = datetime.datetime.now()
    #['HFTrainer','CustomTrain', 'NoTrain']
    if args.trainmode == 'HFTrainer':
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=args.per_device_train_batch_size, #16,
            gradient_accumulation_steps=args.gradient_accumulation_steps, #4,
            per_device_eval_batch_size=args.per_device_eval_batch_size, #16,
            num_train_epochs=args.num_train_epochs, #3,
            #warmup_ratio=args.warmup_ratio, #0.1,
            warmup_steps=args.num_warmup_steps, #500,
            logging_steps=100,
            load_best_model_at_end=True,
            #metric_for_best_model="accuracy",
            #fp16=args.use_fp16,
            push_to_hub=False,
        )
        if args.task == "object-detection":
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=collate_fn,
                train_dataset=train_dataset,
                tokenizer=image_processor,
            )
        else:
            # Initialize our trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=metriceval.compute_metrics,
                tokenizer=image_processor,
                data_collator=collate_fn,
            )
        from DeepDataMiningLearning.hfaudio.hfmodels import load_hfcheckpoint
        checkpoint = load_hfcheckpoint(args.resume_from_checkpoint)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        #trainer.push_to_hub()
    else:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size)
        eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size)

        test_data = next(iter(eval_dataloader))
        print(test_data.keys()) #['pixel_values', 'pixel_mask', 'labels'] 'labels' is list of dicts
        print(test_data["pixel_values"].shape) #[1, 3, 800, 1066]
        print(test_data["pixel_mask"].shape) #[1, 800, 1066]
        if args.trainmode == 'CustomTrain':
            custom_train(args, model, image_processor, train_dataloader, eval_dataloader, metriceval, device, accelerator, do_evaluate=False)
        else:
            evaluate_dataset(model, eval_dataloader, args.task, metriceval, device, image_processor=image_processor, accelerator=accelerator)
    

    # using now() to get current time
    current_time = datetime.datetime.now()
    # Printing value of now.
    print("Starting is:", starting_time)
    print("Time now is:", current_time)
    time_difference = current_time - starting_time
    print("Time difference:", time_difference)
    print("Finished")

def evaluate_dataset(model, val_dataloader, task, metriceval, device, image_processor=None, accelerator=None):
    
    model = model.eval().to(device)
    for step, batch in enumerate(tqdm(val_dataloader)):
        pixel_values = batch["pixel_values"].to(device)#[8, 3, 840, 1333]
        pixel_mask = batch["pixel_mask"].to(device)#[8, 840, 1333]
        #batch = {k: v.to(device) for k, v in batch.items()}
        #"pixel_values" [8, 3, 840, 1333]
        #"pixel_mask" [8, 840, 1333]
        # "labels" 
        with torch.no_grad():
            #outputs = model(**batch)
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask) #DetrObjectDetectionOutput
        
        if task == "image-classification":
            predictions = outputs.logits.argmax(dim=-1)
            if accelerator is not None:
                predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            else:
                predictions, references = predictions, batch["labels"]
        elif task == "object-detection":
            references = [
                {k: v for k, v in t.items()} for t in batch["labels"]
            ]  #resized + normalized, list of dicts ['size', 'image_id', 'boxes'[8,4], 'class_labels', 'area', 'orig_size'[size[2]]]
            orig_target_sizes = torch.stack([target["orig_size"] for target in references], dim=0) #[8,2] shape
            # convert outputs of model to COCO api, list of dicts
            predictions = image_processor.post_process_object_detection(outputs,  threshold=0.0, target_sizes=orig_target_sizes) 
            #list of dicts ['scores', 'labels'[100], 'boxes'(100,4)]
            
            id2label = model.config.id2label 
            #print(batch["labels"][0].keys()) #['size', 'image_id', 'class_labels', 'boxes', 'area', 'iscrowd', 'orig_size']
            image = pixel_values2img(pixel_values)
            pred_boxes = outputs['pred_boxes'].cpu().squeeze(dim=0).numpy() #(100,4) normalized (center_x, center_y, width, height)
            prob = nn.functional.softmax(outputs['logits'], -1) #[1, 100, 92]
            scores, labels = prob[..., :-1].max(-1) #[1, 100] [1, 100]
            scores = scores.cpu().squeeze(dim=0).numpy() #(100,)
            labels = labels.cpu().squeeze(dim=0).numpy() #(100,)
            draw_objectdetection_predboxes(image.copy(), pred_boxes, scores, labels, id2label) #DetrObjectDetectionOutput
            #print(batch["labels"])#list of dicts
            
            #the image size is the not correct
            draw_objectdetection_results(image, predictions[0], id2label)
        metriceval.add_batch(
            predictions=predictions,
            references=references,
        )
        del batch

    eval_metric = metriceval.compute()#metric.compute()
    #print(eval_metric)
    # Printing key-value pairs as tuples
    print("Eval metric Key-Value Pairs:", list(eval_metric.items()))
    return eval_metric

# from huggingface_hub import login
# login()
#huggingface-cli login
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Transformers model on an image classification dataset")
    parser.add_argument('--traintag', type=str, default="hfimage0309",
                    help='Name the current training')
    parser.add_argument('--hubname', type=str, default="detr-resnet-50_finetuned_coco",
                    help='Name the share name in huggingface hub')
    # parser.add_argument(
    #     "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    # )
    parser.add_argument('--trainmode', default="NoTrain", choices=['HFTrainer','CustomTrain', 'NoTrain'], help='Training mode')
    #vocab_path
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="lkk688/detr-resnet-50_finetuned_cppe5", #"emre/detr-resnet-50_finetuned_cppe5", #"output/cppe-5_hfimage0306/epoch_18",#,
        help="Path to pretrained model or model identifier from huggingface.co/models: facebook/detr-resnet-50, google/vit-base-patch16-224-in21k, ",
    )
    parser.add_argument('--usehpc', default=True, action='store_true',
                    help='Use HPC')
    parser.add_argument('--data_path', type=str, default="", help='Huggingface data cache folder') #r"D:\Cache\huggingface", "/data/cmpe249-fa23/Huggingfacecache" "/DATA10T/Cache"
    parser.add_argument('--useamp', default=True, action='store_true',
                    help='Use pytorch amp in training')
    parser.add_argument('--gpuid', default=0, type=int, help='GPU id')
    parser.add_argument('--task', type=str, default="object-detection",
                    help='tasks: image-classification, object-detection')
    parser.add_argument('--data_name', type=str, default="cppe-5",
                    help='data name: detection-datasets/coco, food101, beans, cats_vs_dogs,cppe-5')
    parser.add_argument('--datasplit', type=str, default='train',
                    help='dataset split name in huggingface dataset')
    parser.add_argument('--datatype', type=str, default='huggingface',
                    help='Data type: huggingface, torch')
    #format #"coco": [x_min, y_min, width, height] in pixels 
    #pascal_voc: [x_min, y_min, x_max, y_max] in pixels
    #albumentations  [x_min, y_min, x_max, y_max] normalized
    #yolo: [x_center, y_center, width, height] normalized
    #torchvision 'xyxy' box_convert ['xyxy', 'xywh', 'cxcywh']
    parser.add_argument('--format', type=str, default='coco',
                    help='dataset bbox format: pascal_voc, coco')
    parser.add_argument("--train_dir", type=str, default=None, help="A folder containing the training data.")
    parser.add_argument("--validation_dir", type=str, default=None, help="A folder containing the validation data.")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=-1, #means all data 2000,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.15,
        help="Percent to split off of train for validation",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default="./output", help="Where to store the final model.")
    #parser.add_argument('--outputdir', type=str, default="/data/rnd-liu/output/", help='output path')
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default='epoch',
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--saving_everynsteps",
        type=int,
        default=2,
        help="Save everying 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--image_column_name",
        type=str,
        default="image",
        help="The name of the dataset column containing the image data. Defaults to 'image'.",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default="labels",
        help="The name of the dataset column containing the labels. Defaults to 'label'.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.data_name is None and args.train_dir is None and args.validation_dir is None:
        raise ValueError("Need either a dataset name or a training/validation folder.")

    if args.push_to_hub or args.with_tracking:
        if args.output_dir is None:
            raise ValueError(
                "Need an `output_dir` to create a repo when `--push_to_hub` or `with_tracking` is specified."
            )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

if __name__ == "__main__":
    #"nielsr/convnext-tiny-finetuned-eurostat"
    #"google/bit-50"
    #"microsoft/resnet-50"
    #inference()
    trainmain()

r"""
References: 
https://huggingface.co/docs/transformers/main/en/tasks/image_classification
https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification

"""