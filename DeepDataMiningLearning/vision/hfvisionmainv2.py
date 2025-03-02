
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
from functools import partial
from pprint import pprint
import requests
from PIL import Image, ImageDraw
import io
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import classification_report #pip install scikit-learn
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification, \
    AutoModelForDepthEstimation, AutoModelForObjectDetection, SchedulerType, get_scheduler
from transformers import DefaultDataCollator, Trainer, TrainingArguments
from time import perf_counter
from DeepDataMiningLearning.Utils.visionutil import get_device, saveargs2file, load_ImageNetlabels, read_image
import requests
import cv2
import albumentations#pip install albumentations
from DeepDataMiningLearning.detection.dataset_hf import HFCOCODataset, check_boxsize
from DeepDataMiningLearning.detection.plotutils import draw2pil, pixel_values2img, draw_objectdetection_predboxes, draw_objectdetection_results
#from DeepDataMiningLearning.hfaudio.hfmodels import load_hfcheckpoint
from DeepDataMiningLearning.vision.util import readimg2PILRGB, load_hfcheckpoint, \
    load_config_relativefolder, get_data_transform, get_collate_fn, compute_mapmetrics
from DeepDataMiningLearning.vision.evaluate import myEvaluator, evaluate_dataset
from DeepDataMiningLearning.vision.customtrain import custom_train

logger = get_logger(__name__)


#The PYTORCH_USE_CUDA_DSA environment variable is used to enable the use of the CUDA Direct Storage API (DSA) in PyTorch. DSA is a new API that allows PyTorch to directly access data on the GPU without having to copy it to the CPU first.
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"



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
        #print("classlabel:", classlabel)
        if hasattr(classlabel, "names"):
            labels = classlabel.names
        else:
            labels = split_datasets["train"].unique(label_column_name)
        print(f'Number of Labels: {len(labels)}, all labels: {labels}')
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
    return split_datasets, labels, id2label, label2id, image_column_name, label_column_name


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


def dataset_preprocessing(image_processor, task, label2id=None, format='coco', image_column_name='image', label_column_name='labels'):
    if task =="image-classification":
        train_transforms, val_transforms = get_data_transform(image_processor=image_processor, has_boundingbox=False)
        def preprocess_train(example_batch):
            """Apply _train_transforms across a batch."""
            processed_images = readimg2PILRGB(example_batch[image_column_name])
            #PIL image to RGB
            # example_batch["pixel_values"] = [
            #     train_transforms(image.convert("RGB")) for image in example_batch[image_column_name]
            # ]
            example_batch["pixel_values"] = [
                train_transforms(image) for image in processed_images
            ]
            if label2id is not None and (not isinstance(example_batch[label_column_name][0], int)):
                example_batch[label_column_name] = [int(label2id[y]) for y in example_batch[label_column_name]]
            del example_batch[image_column_name]
            return example_batch

        def preprocess_val(example_batch):
            """Apply _val_transforms across a batch."""
            processed_images = readimg2PILRGB(example_batch[image_column_name])
            # example_batch["pixel_values"] = [
            #     val_transforms(image.convert("RGB")) for image in example_batch[image_column_name]
            # ]
            example_batch["pixel_values"] = [
                val_transforms(image) for image in processed_images
            ]
            if label2id is not None and (not isinstance(example_batch[label_column_name][0], int)):
                example_batch[label_column_name] = [int(label2id[y]) for y in example_batch[label_column_name]]
            del example_batch[image_column_name]
            return example_batch
    
    elif task =="object-detection":
        train_transforms, val_transforms = get_data_transform(image_processor=image_processor, has_boundingbox=True)
        #The image_processor expects the annotations to be in the following format: {'image_id': int, 'annotations': List[Dict]}, 
        #where each dictionary is a COCO object annotation.
        # transforming a batch
        def preprocess_train(examples):#can handle batch
            image_ids = examples["image_id"]
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
                #image = np.array(image.convert("RGB"))[:, :, ::-1] #(720, 1280, 3)HWC
                image = np.array(image.convert("RGB"))
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

#tasks: "depth-estimation", "image-classification", "object-detection"
def load_visionmodel(model_name_or_path, task="image-classification", load_only=True, labels=None, image_maxsize=None, trust_remote_code=True):
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
            #cache_dir=mycache_dir,  
            finetuning_task=task, #"image-classification",
            trust_remote_code=trust_remote_code,
        )

    if image_maxsize is None:
        image_processor = AutoImageProcessor.from_pretrained(
            model_name_or_path,
            #cache_dir=mycache_dir,
            trust_remote_code=trust_remote_code,
        )
    else:
        MAX_SIZE = max(image_maxsize, 28)
        image_processor = AutoImageProcessor.from_pretrained(
            model_name_or_path,
            do_resize=True,
            size={"max_height": MAX_SIZE, "max_width": MAX_SIZE},
            do_pad=True,
            pad_size={"height": MAX_SIZE, "width": MAX_SIZE},
        )
        
    print("Config:", config)
    if task == "image-classification":
        model = AutoModelForImageClassification.from_pretrained(
            model_name_or_path,
            config=config,
            #cache_dir=mycache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            trust_remote_code=trust_remote_code,
        )
    elif task == "depth-estimation":
        model = AutoModelForDepthEstimation.from_pretrained(
            model_name_or_path, 
            config=config,
            #cache_dir=mycache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            trust_remote_code=trust_remote_code,
        )
    elif task == "object-detection":
        model = AutoModelForObjectDetection.from_pretrained(
            model_name_or_path, 
            config=config,
            #cache_dir=mycache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            trust_remote_code=trust_remote_code,
        )
    #model.config.id2label
    return model, image_processor


def trainmain():
    args = parse_args()

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
        dataset, labels, id2label, label2id, args.image_column_name, args.label_column_name \
            = load_visiondataset(data_name=args.data_name, \
                split=args.datasplit, train_dir=args.train_dir, validation_dir=args.validation_dir, \
                task=args.task, format=args.format, max_train_samples = args.max_train_samples, train_val_split=args.train_val_split, \
                image_column_name=args.image_column_name, label_column_name=args.label_column_name, mycache_dir=mycache_dir)

    # Load pretrained model and image processor
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    image_maxsize = args.image_maxsize if hasattr(args, "image_maxsize") else None
    model, image_processor = load_visionmodel(args.model_name_or_path, task=args.task, load_only=False, labels=labels, image_maxsize=image_maxsize, trust_remote_code=True)
    model.config.id2label = id2label

    # Preprocessing the datasets
    # Define torchvision transforms to be applied to each image.

    with accelerator.main_process_first():
        preprocess_train, preprocess_val = dataset_preprocessing(image_processor=image_processor, task=args.task, label2id=label2id, format=args.format, image_column_name=args.image_column_name, label_column_name=args.label_column_name)
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

    collate_fn = get_collate_fn(args.task, image_processor, args.label_column_name, require_padding=False)

    metriceval = myEvaluator(task=args.task, useHFevaluator=args.useHFevaluator, dualevaluator=False, \
                            processor=image_processor, coco=coco, mycache_dir=mycache_dir)

    
    # using now() to get current time
    starting_time = datetime.datetime.now()
    #['HFTrainer','CustomTrain', 'NoTrain']
    if args.trainmode == 'HFTrainer':
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            remove_unused_columns=False,
            eval_strategy="epoch", #"no", #,
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
            eval_compute_metrics_fn = partial(
                compute_mapmetrics, image_processor=image_processor, id2label=id2label, threshold=0.0
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=collate_fn,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=image_processor,
                compute_metrics=eval_compute_metrics_fn,
            )
        else:
            # Initialize our trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=metriceval.compute_metrics,
                processing_class=image_processor,
                data_collator=collate_fn,
            )
        checkpoint = load_hfcheckpoint(args.resume_from_checkpoint)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="test")
        pprint(metrics)
        #trainer.push_to_hub()
    else:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size)
        eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size)

        test_data = next(iter(eval_dataloader))
        print(test_data.keys()) #['pixel_values', 'pixel_mask', 'labels'] 'labels' is list of dicts
        #print(test_data["pixel_values"].shape) #[1, 3, 800, 1066]
        #print(test_data["pixel_mask"].shape) #[1, 800, 1066]
        if args.trainmode == 'CustomTrain':
            custom_train(args, model, image_processor, train_dataloader, eval_dataloader, metriceval, device, accelerator, do_evaluate=True)
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


# from huggingface_hub import login
# login()
#huggingface-cli login
def parse_args():
    parser = argparse.ArgumentParser(description="HuggingFace Vision Training")
    parser.add_argument('--config', type=str, default="classify_pets_vit.yaml",
                    help='Name the current training')
    args = parser.parse_args()
    
    configfile=load_config_relativefolder(args.config)
    for key, value in configfile.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action='store_true' if value else 'store_false', default=value)
        else:
            parser.add_argument(f"--{key}", default=value)

    args = parser.parse_args([]) #parse empty arguments, so the default value will be used.
    # Access properties using args.property_name
    print("Training Configuration:")
    print(f"  Train Tag: {args.traintag}")
    print(f"  Model: {args.model_name_or_path}")
    
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
