
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
import requests
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import classification_report
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification, SchedulerType, get_scheduler
from transformers import DefaultDataCollator, Trainer, TrainingArguments
from time import perf_counter
from DeepDataMiningLearning.visionutil import get_device, saveargs2file, load_ImageNetlabels, read_image
import requests
import cv2

logger = get_logger(__name__)

#The PYTORCH_USE_CUDA_DSA environment variable is used to enable the use of the CUDA Direct Storage API (DSA) in PyTorch. DSA is a new API that allows PyTorch to directly access data on the GPU without having to copy it to the CPU first.
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"

class myEvaluator:
    def __init__(self, task, useHFevaluator=False, dualevaluator=False, labels=None, processor=None, mycache_dir=None):
        print("useHFevaluator:", useHFevaluator)
        print("dualevaluator:", dualevaluator)
        self.useHFevaluator = useHFevaluator
        self.dualevaluator = dualevaluator
        self.task = task
        self.preds = []
        self.refs = []
        self.labels = labels
        self.processor = processor
        self.HFmetric = None
        if self.task == "image-classification":
            self.metricname = "accuracy" #"mse" "wer"
        else:
            self.metricname = "accuracy"
        self.LOmetric = None
        if self.useHFevaluator:
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
                print("HF evaluator result1:", results)
                results2 = self.HFmetric.compute(predictions=self.preds, references=self.refs) #the same results
                print("HF evaluator result2:", results2)
                if not isinstance(results, dict):
                    #wer output is float, convert to dict
                    results = {self.metricname: results}
            else:
                results = self.mycompute(predictions=self.preds, references=self.refs)
            self.preds.clear()
            self.refs.clear()
        return results
    
    def add_batch(self, predictions, references):
        if self.useHFevaluator == True:
            self.HFmetric.add_batch(predictions=predictions, references=references)
        #self.preds.append(predictions)
        self.refs.extend(references)
        self.preds.extend(predictions)
        #references: list of list
        # for ref in references:
        #     self.refs.append(ref[0])
        #print(len(self.refs))

# from huggingface_hub import login
# login()
#huggingface-cli login
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Transformers model on an image classification dataset")
    parser.add_argument('--traintag', type=str, default="hfimage0226",
                    help='Name the current training')
    parser.add_argument('--trainmode', default="CustomTrain", choices=['HFTrainer','CustomTrain', 'NoTrain'], help='Training mode')
    #vocab_path
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="google/vit-base-patch16-224-in21k",
        help="Path to pretrained model or model identifier from huggingface.co/models: google/vit-base-patch16-224-in21k, ",
    )
    parser.add_argument('--usehpc', default=True, action='store_true',
                    help='Use HPC')
    parser.add_argument('--data_path', type=str, default="", help='Huggingface data cache folder') #r"D:\Cache\huggingface", "/data/cmpe249-fa23/Huggingfacecache" "/DATA10T/Cache"
    parser.add_argument('--useamp', default=True, action='store_true',
                    help='Use pytorch amp in training')
    parser.add_argument('--gpuid', default=0, type=int, help='GPU id')
    parser.add_argument('--task', type=str, default="image-classification",
                    help='tasks: image-classification')
    parser.add_argument('--data_name', type=str, default="cats_vs_dogs",
                    help='data name: food101, beans')
    parser.add_argument('--datasplit', type=str, default='train',
                    help='dataset split name in huggingface dataset')
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
        default=16,
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
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
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
def load_visiondataset(data_name=None, split="train", train_dir=None, validation_dir=None, max_train_samples = 2000, train_val_split=0.15, \
                       image_column_name='image', label_column_name='labels', mycache_dir=None):
    if data_name is not None:
        if max_train_samples and max_train_samples>0 and split is not None:
            data_split=f"{split}[:{max_train_samples}]" #"train+validation"
        elif split is not None:
            data_split=f"{split}"
        else:
            data_split=None
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(data_name,split=data_split, cache_dir=mycache_dir, ignore_verifications=True, trust_remote_code=True)
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
    #some datset the labels name is different
    if data_name == "food101": #https://huggingface.co/datasets/food101
        image_column_name = "image"
        label_column_name = "label"
    elif 'cats_vs_dogs' in data_name:
        image_column_name = "image"
        label_column_name = "labels"
    elif 'openfire' in data_name:
        image_column_name='image_url'
        label_column_name='is_wildfire'

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
    print(split_datasets["train"][0][label_column_name])
    #The Datasets library is made for processing data very easily. We can write custom functions, 
    #which can then be applied on an entire dataset (either using .map() or .set_transform()).
    return split_datasets, labels, image_column_name, label_column_name

#tasks: "depth-estimation", "image-classification"
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
    #model.config.id2label
    return model, image_processor


def custom_train(args, model, image_processor, train_dataset, eval_dataset, collate_fn, metriceval, accelerator=None):
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size)

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
    logger.info(f"  Num examples = {len(train_dataset)}")
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

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            metriceval.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metriceval.compute()#metric.compute()
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

            all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f)

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
        if os.environ['HF_HOME']:
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
        dataset, labels, args.image_column_name, args.label_column_name = load_visiondataset(data_name=args.data_name, \
                                    split=args.datasplit, train_dir=args.train_dir, validation_dir=args.validation_dir, \
                                    max_train_samples = args.max_train_samples, train_val_split=args.train_val_split, \
                                    image_column_name=args.image_column_name, label_column_name=args.label_column_name, mycache_dir=mycache_dir)

    # Load pretrained model and image processor
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    model, image_processor = load_visionmodel(args.model_name_or_path, task=args.task, load_only=False, labels=labels, mycache_dir=mycache_dir, trust_remote_code=True)

    # Preprocessing the datasets
    # Define torchvision transforms to be applied to each image.
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"]) #(224, 224)
    
    normalize = (
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std")
        else Lambda(lambda x: x)
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

    def preprocess_train(example_batch):
        """Apply _train_transforms across a batch."""
        #PIL image to RGB
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch[args.image_column_name]
        ]
        del example_batch[args.image_column_name]
        return example_batch

    def preprocess_val(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [
            val_transforms(image.convert("RGB")) for image in example_batch[args.image_column_name]
        ]
        del example_batch[args.image_column_name]
        return example_batch

    with accelerator.main_process_first():
        #The transforms are applied on the fly when you load an element of the dataset:
        train_dataset = dataset["train"].with_transform(preprocess_train)
        eval_dataset = dataset[valkey].with_transform(preprocess_val)

        #train_dataset = dataset["train"].map(preprocess_train)
        #eval_dataset = dataset[valkey].map(preprocess_val)

    # DataLoaders creation:
    #used to batch examples together. Each batch consists of 2 keys, namely pixel_values and labels.
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example[args.label_column_name] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}
    #similar to 
    #collate_fn = DefaultDataCollator()

    metriceval = myEvaluator(task=args.task, useHFevaluator=True, dualevaluator=False, \
                            labels=labels, processor=image_processor, mycache_dir=mycache_dir)

 
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
    elif args.trainmode == 'CustomTrain':
        custom_train(args, model, image_processor, train_dataset, eval_dataset, collate_fn, metriceval, accelerator)

    # using now() to get current time
    current_time = datetime.datetime.now()
    # Printing value of now.
    print("Starting is:", starting_time)
    print("Time now is:", current_time)
    time_difference = current_time - starting_time
    print("Time difference:", time_difference)
    print("Finished")


#tasks: "depth-estimation", "image-classification"
class MyVisionInference():
    def __init__(self, model_name, model_type="huggingface", task="image-classification", cache_dir="./output", gpuid='0') -> None:
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.model_type = model_type
        #self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device, useamp = get_device(gpuid=gpuid, useamp=False)
        self.task = task
        self.model_name = model_name
        self.model = None
        self.image_processor = None
        self.transforms = None
        if isinstance(model_name, str) and model_type=="huggingface":
            os.environ['HF_HOME'] = cache_dir #'~/.cache/huggingface/'
            self.model, self.image_processor = load_visionmodel(model_name, task=task, load_only=True, labels=None, mycache_dir=cache_dir, trust_remote_code=True)
        # elif isinstance(model_name, torch.nn.Module):
        #     self.model = model_name
        elif isinstance(model_name, str) and task=="image-classification":#torch model
            self.model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True) #'resnet18'
        elif isinstance(model_name, str) and task=="depth-estimation":#torch model
            #https://pytorch.org/hub/intelisl_midas_v2/
            #model_names: "MiDaS_small", "DPT_Hybrid", "DPT_Large"
            self.model = torch.hub.load('intel-isl/MiDaS', model_name, pretrained=True)
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transforms = transforms.dpt_transform #.small_transform resize(384,) Normalized

        #get labels for classification
        if task=="image-classification" and model_type=="huggingface":
            self.id2label = self.model.config.id2label
        elif task=="image-classification":
            labels=load_ImageNetlabels(filepath='sampledata/imagenet_labels.txt')
            self.id2label = {str(i): label for i, label in enumerate(labels)}
        elif task=="depth-estimation":
            pass

        self.model=self.model.to(self.device)
        self.model.eval()
    
    def mypreprocess(self,inp):
        #https://pytorch.org/vision/stable/transforms.html
        #transforms.ToTensor()
        #Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        manual_transforms = transforms.Compose([
            #transforms.Resize((224, 224)), # 1. Reshape all images to 224x224 (though some models may require different sizes)
            transforms.ToTensor(), # 2. changes the input arrays from HWC to CHW, Turn image values to between 0 & 1 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                                std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
        ])
        if self.transforms is not None:
            inp = self.transforms(inp)#.unsqueeze(0) already added one dimension
        else:
            #inp = transforms.ToTensor()(inp).unsqueeze(0)
            inp = manual_transforms(inp).unsqueeze(0)
        return inp

    def __call__(self, image):
        self.image = read_image(image, use_pil=True, use_cv2=False, output_format='numpy', plotfig=False)
        #HWC numpy (427, 640, 3)
        if self.model_type=="huggingface":
            inputs = self.image_processor(self.image, return_tensors="pt").pixel_values
            print(inputs.shape) #torch.Size([1, 3, 224, 224]) [1, 3, 350, 518]
        else:
            inputs = self.mypreprocess(self.image)
            print(inputs.shape) #torch.Size([1, 3, 384, 576])
        inputs = inputs.to(self.device)

        start_time = perf_counter()
        with torch.no_grad():
            outputs = self.model(inputs)
        end_time = perf_counter()
        print(f'Elapsed inference time: {end_time-start_time:.3f}s')
        
        results = None
        if self.task=="image-classification":
            results = self.classification_postprocessing(outputs) #return confidence dicts
        elif self.task=="depth-estimation":
            results = self.depth_postprocessing(outputs, recolor=True) #return PIL image
        return results
    
    def depth_postprocessing(self, outputs, recolor=True):
        if self.model_type=="huggingface":
            predicted_depth = outputs.predicted_depth #[1, 416, 640], [1, 384, 384]
        else:
            predicted_depth  = outputs
        # interpolate to original size
        #The input dimensions are interpreted in the form: mini-batch x channels x [optional depth] x [optional height] x width.
        predicted_depth_input = predicted_depth.unsqueeze(1) #[1, 1, 384, 576]

        #read the height/width from image's shape
        if isinstance(self.image, Image.Image):
            self.height = self.image.height
            self.width = self.image.width
            target_size = self.image.size[::-1]#(width, height) (W=640, H=427)->(H=427, W=640)
        elif isinstance(self.image, np.ndarray): #HWC format
            self.height = self.image.shape[0]
            self.width = self.image.shape[1]
            target_size = (self.height, self.width) #(427, 640)
        
        #The input dimensions are interpreted in the form: mini-batch x channels x [optional depth] x [optional height] x width.
        prediction = torch.nn.functional.interpolate(
            predicted_depth_input,
            size=target_size,#(H=427, W=640)
            mode="bicubic", #'bilinear', uses a bicubic interpolation algorithm to compute the values of the new tensor
            align_corners=False,
        ) #[1, 1, 427, 640]
        #output = prediction.squeeze().cpu().numpy() #[427, 640]
        #formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = (prediction - prediction.min()) / (prediction.max() - prediction.min()) * 255.0
        formatted = depth.squeeze().cpu().numpy().astype(np.uint8)
        print(formatted.shape) #(427, 640)
        if recolor:
            #Recolor the depth map from grayscale to colored
            #normalized_image = image.astype(np.uint8)
            formatted = cv2.applyColorMap(formatted, cv2.COLORMAP_HOT)[:, :, ::-1]
            print(formatted.shape) #(427, 640, 3)
    
        depth = Image.fromarray(formatted)
        #depth.show()
        depth.save("data/depth_testresult.jpg") 
        return depth

    def classification_postprocessing(self, outputs):
        if self.model_type=="huggingface":
            logits = outputs.logits #torch.Size([1, 1000])
        else:
            logits = outputs
        predictions = torch.nn.functional.softmax(logits[0], dim=0) #[1000]
        #print(predictions.shape) #torch.Size([1000])
        predictions = predictions.cpu().numpy()
        confidences = {self.id2label[i]: float(predictions[i]) for i in range(len(self.id2label))} #len=999

        predmax_idx = np.argmax(predictions, axis=-1)
        predmax_label = self.id2label[predmax_idx]
        predmax_confidence = float(predictions[predmax_idx])
        print(f"predmax_idx: {predmax_idx}, predmax_label: {predmax_label}, predmax_confidence: {predmax_confidence}")
        return confidences

def vision_inferencetest(model_name_or_path, task="image-classification", mycache_dir=None):

    url = 'https://huggingface.co/nielsr/convnext-tiny-finetuned-eurostat/resolve/main/forest.png'
    image = Image.open(requests.get(url, stream=True).raw)
    #show the PIL image
    #image.show()
    #load the model
    dataset = load_dataset("huggingface/cats-image", cache_dir=mycache_dir)
    image = dataset["test"]["image"][0]
    print(len(dataset["test"]["image"]))
    im1 = image.save("test.jpg") 

    myinference = MyVisionInference(model_name=model_name_or_path, task=task, model_type="huggingface", cache_dir=mycache_dir)
    confidences = myinference(image)
    print(confidences)

    #model_name_or_path = "nielsr/convnext-tiny-finetuned-eurostat"
    #task = "image-classification"
    model, image_processor = load_visionmodel(model_name_or_path, task=task, load_only=True, labels=None, mycache_dir=mycache_dir, trust_remote_code=True)
    id2label = model.config.id2label
    #model = AutoModelForImageClassification.from_pretrained("nielsr/convnext-tiny-finetuned-eurostat")
    #load the image processor
    #image_processor = AutoImageProcessor.from_pretrained("nielsr/convnext-tiny-finetuned-eurostat")
    #preprocess the image (PIL)
    inputs = image_processor(image.convert("RGB"), return_tensors="pt")
    print(inputs.pixel_values.shape) #torch.Size([1, 3, 224, 224])
    #inference
    with torch.no_grad():
        outputs = model(**inputs) #key="logits", "loss"
    #get the prediction
    logits = outputs.logits #torch.Size([1, 1000])

    predictions = torch.nn.functional.softmax(logits[0], dim=0) #[1000]
    confidences = {id2label[i]: float(predictions[i]) for i in range(len(id2label))} #len=999

    pred = logits.argmax(dim=-1) #torch.Size([1])
    #print the prediction
    print(pred[0]) #tensor(646)
    predicted_class_idx = pred[0].item() #1
    print("Predicted class:", id2label[predicted_class_idx])
    return confidences

#https://huggingface.co/docs/transformers/main/en/tasks/zero_shot_image_classification
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
def clip_test(mycache_dir=None):
    #url = "https://unsplash.com/photos/g8oS8-82DxI/download?ixid=MnwxMjA3fDB8MXx0b3BpY3x8SnBnNktpZGwtSGt8fHx8fDJ8fDE2NzgxMDYwODc&force=true&w=640"
    #candidate_labels = ["fox", "bear", "seagull", "owl"]
    
    url = "https://unsplash.com/photos/xBRQfR2bqNI/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjc4Mzg4ODEx&force=true&w=640" #car
    candidate_labels = ["tree", "car", "bike", "cat"]
    image = Image.open(requests.get(url, stream=True).raw)
    checkpoint = "openai/clip-vit-large-patch14"
    model = AutoModelForZeroShotImageClassification.from_pretrained(checkpoint, cache_dir=mycache_dir)
    processor = AutoProcessor.from_pretrained(checkpoint, cache_dir=mycache_dir)

    
    inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True) 
    #'input_ids'[4, 3], 'pixel_values'[1, 3, 224, 224], 'attention_mask'[4, 3]

    with torch.no_grad():
        outputs = model(**inputs) #CLIPOutput(loss=None, logits_per_image[1,4], logits_per_text[4,1]

    logits = outputs.logits_per_image[0] #[4]
    probs = logits.softmax(dim=-1).numpy()
    scores = probs.tolist()

    result = [
        {"score": score, "label": candidate_label}
        for score, candidate_label in sorted(zip(probs, candidate_labels), key=lambda x: -x[0])
    ]

    print(result)

#https://huggingface.co/docs/transformers/main/en/tasks/monocular_depth_estimation
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
def depth_test(mycache_dir=None):
    url = "https://unsplash.com/photos/HwBAsSbPBDU/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MzR8fGNhciUyMGluJTIwdGhlJTIwc3RyZWV0fGVufDB8MHx8fDE2Nzg5MDEwODg&force=true&w=640"
    image = Image.open(requests.get(url, stream=True).raw)
    #image.size()#The size attribute returns a (width, height) tuple
    image.save("data/depth_test.jpg") 

    checkpoint = "Intel/dpt-large" #"LiheYoung/depth-anything-base-hf" #"Intel/dpt-large" #"vinvino02/glpn-nyu"
    depthinference = MyVisionInference(model_name=checkpoint, model_type="huggingface", task="depth-estimation", cache_dir=mycache_dir)
    #checkpoint = "DPT_Large" #pip install timm
    #depthinference = MyVisionInference(model_name=checkpoint, model_type="torch", task="depth-estimation", cache_dir=mycache_dir)
    results = depthinference(image=image)
    
    
    image_processor = AutoImageProcessor.from_pretrained(checkpoint, cache_dir=mycache_dir)
    model = AutoModelForDepthEstimation.from_pretrained(checkpoint, cache_dir=mycache_dir)

    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = model(pixel_values) #only predicted_depth key
        predicted_depth = outputs.predicted_depth #[1, 416, 640], [1, 384, 384]
    
    # interpolate to original size
    #The input dimensions are interpreted in the form: mini-batch x channels x [optional depth] x [optional height] x width.
    predicted_depth_input = predicted_depth.unsqueeze(1) #[1, 1, 384, 384]
    target_size = image.size[::-1]#(width, height) (W=640, H=427)->(H=427, W=640)
    prediction = torch.nn.functional.interpolate(
        predicted_depth_input,
        size=target_size,#(H=427, W=640)
        mode="bicubic", #uses a bicubic interpolation algorithm to compute the values of the new tensor
        align_corners=False,
    )
    output = prediction.squeeze().numpy() #[427, 640]
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    #depth.show()
    depth.save("data/depth_testresult.jpg") 

def inference():
    mycache_dir="/home/lkk/Developer/"#r"D:\Cache\huggingface"
    #os.environ['HF_HOME'] = mycache_dir #'~/.cache/huggingface/'
    mycache_dir = os.environ['HF_HOME']
    depth_test(mycache_dir=mycache_dir)
    clip_test(mycache_dir=mycache_dir)
    confidences = vision_inferencetest(model_name_or_path="google/bit-50", task="image-classification", mycache_dir=mycache_dir)

if __name__ == "__main__":
    #"nielsr/convnext-tiny-finetuned-eurostat"
    #"google/bit-50"
    #"microsoft/resnet-50"
    trainmain()

r"""
References: 
https://huggingface.co/docs/transformers/main/en/tasks/image_classification
https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification

"""