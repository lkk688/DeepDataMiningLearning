import datetime
import os
import time
import math
import sys
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection

from DeepDataMiningLearning.detection import utils
from DeepDataMiningLearning.detection.trainutils import create_aspect_ratio_groups, GroupedBatchSampler

from DeepDataMiningLearning.detection.dataset import get_dataset #get_cocodataset, get_kittidataset, get_transform
#pip install torchinfo ultralytics
from DeepDataMiningLearning.detection.models import create_detectionmodel #get_torchvision_detection_models, modify_fasterrcnnheader
from DeepDataMiningLearning.detection.myevaluator import simplemodelevaluate, modelevaluate, simplemodelevaluate_old

try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.") #pip install -q torchinfo


#Select the visible GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3" #"0,1"
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image

# ANSI color codes for cleaner printouts
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"

def visualize_dataset_sample(
    dataset,
    sample_idx=None,
    save_path="sample_visualization.jpg",
    category_names=None,
    box_format="auto",
    verbose=True,
):
    """
    Visualize one sample (image + bounding boxes + labels) from a dataset.
    Automatically detects COCO / VOC / YOLO / custom formats, prints
    detected data structure, and supports external category name mapping.

    Args:
        dataset: PyTorch Dataset instance (e.g., get_coco(), CocoDetection, VOC, YOLO, custom)
        sample_idx (int or None): which sample to visualize (random if None)
        save_path (str): output image path
        category_names (dict or list or None): optional mapping from label_id → class name
        box_format (str): 'xyxy', 'xywh', or 'auto'
        verbose (bool): whether to print detailed dataset structure
    """
    # ------------------------------
    # 1. Select a random sample
    # ------------------------------
    if sample_idx is None:
        sample_idx = random.randint(0, len(dataset) - 1)

    sample = dataset[sample_idx]
    if not (isinstance(sample, (list, tuple)) and len(sample) == 2):
        raise TypeError(f"{RED}Dataset sample must be (image, target), got {type(sample)}{RESET}")
    img, target = sample

    # ------------------------------
    # 2. Convert image to PIL for visualization
    # ------------------------------
    if torch.is_tensor(img):
        img = to_pil_image(img)
    width, height = img.size

    if verbose:
        print(f"\n{CYAN}=== Dataset Sample Info ==={RESET}")
        print(f"Index: {sample_idx}")
        print(f"Image size: {width}x{height}")
        print(f"Image type: {type(img)}")
        print(f"Target type: {type(target)}")
        if isinstance(target, dict):
            print(f"Target keys: {list(target.keys())}")
        elif isinstance(target, list) and len(target) > 0:
            print(f"List of dicts, keys example: {list(target[0].keys())}")

    # ------------------------------
    # 3. Determine category name mapping
    # ------------------------------
    cat_map = None
    if category_names is not None:
        if isinstance(category_names, list):
            cat_map = {i: name for i, name in enumerate(category_names)}
        elif isinstance(category_names, dict):
            cat_map = category_names
        else:
            raise TypeError(f"{RED}category_names must be a list or dict{RESET}")
        if verbose:
            print(f"{GREEN}Using external category_names mapping with {len(cat_map)} entries.{RESET}")

    elif hasattr(dataset, "coco"):
        try:
            categories = dataset.coco.loadCats(dataset.coco.getCatIds())
            cat_map = {c["id"]: c["name"] for c in categories}
            if verbose:
                print(f"{GREEN}Detected COCO dataset: loaded {len(cat_map)} category names.{RESET}")
        except Exception as e:
            if verbose:
                print(f"{YELLOW}Warning: Could not load COCO category names: {e}{RESET}")

    else:
        if verbose:
            print(f"{YELLOW}No category name mapping detected (labels will be numeric only).{RESET}")

    # ------------------------------
    # 4. Extract bounding boxes and labels
    # ------------------------------
    boxes, labels = [], []
    dataset_type = "unknown"

    # COCO-style list[dict]
    if isinstance(target, list) and len(target) > 0 and isinstance(target[0], dict):
        dataset_type = "COCO (list[dict])"
        for ann in target:
            bbox = ann.get("bbox", None)
            if bbox is None:
                continue
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])  # xywh → xyxy
            labels.append(ann.get("category_id", 0))

    # dict-style dataset (get_coco, custom)
    elif isinstance(target, dict):
        dataset_type = "Custom dict-based dataset"
        if "boxes" in target:
            boxes = target["boxes"].tolist() if torch.is_tensor(target["boxes"]) else target["boxes"]
        if "labels" in target:
            labels = target["labels"].tolist() if torch.is_tensor(target["labels"]) else target["labels"]
        elif "classes" in target:
            labels = target["classes"]

    # YOLO tensor format: [cls, x_center, y_center, w, h]
    elif isinstance(target, torch.Tensor) and target.ndim == 2 and target.shape[1] >= 5:
        dataset_type = "YOLO tensor format"
        labels = target[:, 0].tolist()
        boxes_xywh = target[:, 1:5].tolist()
        boxes = [[x - w / 2, y - h / 2, x + w / 2, y + h / 2] for x, y, w, h in boxes_xywh]

    if verbose:
        print(f"{CYAN}Detected dataset format: {dataset_type}{RESET}")
        print(f"Number of boxes: {len(boxes)}")
        if len(labels) > 0:
            print(f"Label sample: {labels[:5]}")

    if len(boxes) == 0:
        print(f"{YELLOW}[Warning] No boxes found for sample {sample_idx}.{RESET}")
        return

    # ------------------------------
    # 5. Auto-detect box format (xyxy vs xywh)
    # ------------------------------
    if box_format == "auto" and len(boxes) > 0:
        x1, y1, x2, y2 = boxes[0]
        if x2 < x1 or y2 < y1:
            boxes = [[x, y, x + w, y + h] for (x, y, w, h) in boxes]
            print(f"{YELLOW}Auto-detected COCO xywh boxes; converted to xyxy format.{RESET}")
        else:
            print(f"{GREEN}Detected xyxy format boxes (no conversion needed).{RESET}")
            
    # ------------------------------
    # 6. Draw boxes and save visualization
    # ------------------------------
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    ax.set_title(f"Sample {sample_idx} — {len(boxes)} objects", fontsize=14)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = (random.random(), random.random(), random.random())
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        label_id = int(labels[i]) if i < len(labels) else -1
        label_text = str(label_id)
        if cat_map and label_id in cat_map:
            label_text = f"{cat_map[label_id]} ({label_id})"

        ax.text(
            x1, max(y1 - 5, 5),
            label_text,
            color="white", fontsize=10, weight="bold",
            bbox=dict(facecolor=color, alpha=0.6, pad=2, edgecolor="none")
        )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if verbose:
        print(f"{GREEN}[INFO] Saved visualization to: {save_path}{RESET}")
    plt.close(fig)

MACHINENAME='HPC'
USE_AMP=True #AUTOMATIC MIXED PRECISION
# if MACHINENAME=='HPC':
#     os.environ['TORCH_HOME'] = '/data/cmpe249-fa23/torchhome/'
#     DATAPATH='/data/cmpe249-fa23/torchvisiondata/'
# elif MACHINENAME=='Container':
#     os.environ['TORCH_HOME'] = '/data/torchhome/'
#     DATAPATH='/data/torchvisiondata'
# else:
#     DATAPATH='./data'

#dataset: #coco, /mnt/e/Shared/Dataset/coco2017
#kitti /data/cmpe249-fa23/torchvisiondata/Kitti/

#waymococo: /mnt/e/Shared/Dataset/waymodata/waymo_subset_coco_4000step5/
#/mnt/e/Shared/Dataset/waymodata/waymo_subset_coco_4000step5/annotations.json

#(mycondapy310) [010796032@cs003 detection]$ torchrun --nproc_per_node=4 mytrain.py --batch-size=32
def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="/mnt/e/Shared/Dataset/waymodata/waymo_subset_coco_4000step5/", type=str, help="dataset path") #"/data/cmpe249-fa23/WaymoCOCO/"
    parser.add_argument("--annotationfile", default="/mnt/e/Shared/Dataset/waymodata/waymo_subset_coco_4000step5/annotations.json", type=str, help="dataset annotion file path, e.g., coco json file") #annotations_train200new.json
    parser.add_argument(
        "--dataset",
        default="waymococo", #coco, waymococo
        type=str,
        help="dataset name. Use coco for object detection and instance segmentation and coco_kp for Keypoint detection",
    )
    parser.add_argument("--model", default="customrcnn_resnet152", type=str, help="model name") #customrcnn_resnet152, fasterrcnn_resnet50_fpn_v2
    parser.add_argument("--nocustomize", action="store_false", default=True, help="whether change the model header for custom num_classes")
    parser.add_argument("--trainable", default=0, type=int, help="number of trainable layers (sequence) of backbone")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=16, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=60, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--saveeveryepoch", default=4, type=int, metavar="N", help="number of epochs to save")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument(
        "--lr",
        default=0.02,
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    parser.add_argument("--print-freq", default=5, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./output", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint") #/data/cmpe249-fa23/trainoutput/kitti/model_4.pth
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=-1, type=int) #3
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    # parser.add_argument(
    #     "--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone"
    # )
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        default=False,
        action="store_true",
        help="Only test the model",
    )
    parser.add_argument(
        "--debugmode",
        default=False,
        action="store_true",
        help="DEBUG output",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    # distributed training parameters
    parser.add_argument("--multigpu", default=False, type=bool, help="disable torch ddp")
    parser.add_argument("--world-size", default=4, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    parser.add_argument("--expname", default="1005", help="experiment name, create a sub-folder")

    return parser

def main(args):
    if args.backend.lower() == "tv_tensor" and not args.use_v2:
        raise ValueError("Use --use-v2 if you want to use the tv_tensor backend.")
    # if args.dataset not in ("coco", "coco_kp"):
    #     raise ValueError(f"Dataset should be coco or coco_kp, got {args.dataset}")
    if "keypoint" in args.model and args.dataset != "coco_kp":
        raise ValueError("Oops, if you want Keypoint detection, set --dataset coco_kp")
    if args.dataset == "coco_kp" and args.use_v2:
        raise ValueError("KeyPoint detection doesn't support V2 transforms yet")

    if args.output_dir:
        utils.mkdir(args.output_dir)
        if args.expname:
            args.output_dir = os.path.join(args.output_dir, args.dataset, args.expname)
        else:
            args.output_dir = os.path.join(args.output_dir, args.dataset)
        utils.mkdir(args.output_dir)

    if args.multigpu:
        utils.init_distributed_mode(args)
        args.distributed = True
    else:
        args.distributed = False
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset(args.dataset, is_train=True, is_val=False, args=args) #get_dataset
    dataset_test, _ = get_dataset(args.dataset, is_train=False, is_val=True, args=args, img_size=None)

    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    # idxsplit=int(len(indices)*0.80)#159948
    # dataset = torch.utils.data.Subset(dataset, indices[:idxsplit])
    # dataset_test = torch.utils.data.Subset(dataset, indices[idxsplit+1:])
    # dataset_test.transform = get_transform(is_train=False, args=args)
    print("train set len:", len(dataset))
    print("Test set len:", len(dataset_test))
    #visualize_dataset_sample(dataset, sample_idx=0, save_path="datasetcoco_trainsample.jpg")
    visualize_dataset_sample(dataset_test, save_path="datasetcoco_valsample.jpg")

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    train_collate_fn = utils.collate_fn
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=train_collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=1, collate_fn=utils.collate_fn
    )
    # from DeepDataMiningLearning.detection.myevaluator import get_coco_val_dataset
    # dataset_test = get_coco_val_dataset(args.data_path)
    # visualize_dataset_sample(dataset_test, sample_idx=0, save_path="datasetcoco_sample2.jpg")

    # data_loader_test = torch.utils.data.DataLoader(
    #         dataset_test,
    #         batch_size=1,
    #         shuffle=False,
    #         num_workers=4,
    #         collate_fn=lambda x: tuple(zip(*x)),
    #     )

    print("Creating model")
    #kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
    kwargs = {"trainable": args.trainable}
    if args.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    
    model, preprocess, model_classes = create_detectionmodel(args.model, num_classes, customize=args.nocustomize, trainable_layers=args.trainable)
    model.to(device)
    
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.norm_weight_decay is None:
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

    #scaler = torch.cuda.amp.GradScaler() if args.amp else None
    scaler = torch.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        #optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        #torch.backends.cudnn.deterministic = True
        # Load COCO val2017 dataset
        # from torch.utils.data import DataLoader
        # from DeepDataMiningLearning.detection.myevaluator import get_coco_val_dataset
        # dataset = get_coco_val_dataset(args.data_path)
        # #standarded CocoDetection, returns: (image, [annotation_dict_1, annotation_dict_2, ...])
        # data_loader = DataLoader(
        #     dataset,
        #     batch_size=1,
        #     shuffle=False,
        #     num_workers=4,
        #     collate_fn=lambda x: tuple(zip(*x)),
        # )
        #simplemodelevaluate(model, data_loader_test, device=device)
        if hasattr(dataset, "coco_class_map") and (len(model_classes) == 91 or len(model_classes) == 80):
            simplemodelevaluate(model, data_loader_test, device, class_map=dataset.coco_class_map, class_names=dataset.CLASSES, DEBUG=args.debugmode)
        else:
            #simplemodelevaluate(model, data_loader_test, device)
            simplemodelevaluate(model, data_loader_test, device, DEBUG=args.debugmode)
        #evaluate(model, data_loader_test, device=device)
        #evaluate(model, data_loader, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, scaler)
        lr_scheduler.step()
        if epoch % args.saveeveryepoch == 0 or epoch == args.epochs:
            if args.output_dir:
                checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "args": args,
                    "epoch": epoch,
                }
                if args.amp:
                    checkpoint["scaler"] = scaler.state_dict()
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

            # evaluate after current epoch
            #modelevaluate(model, data_loader_test, device=device)
            simplemodelevaluate(model, data_loader_test, device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images) #list of [3, 1280, 1920]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets] #tuple to list
        #with torch.cuda.amp.autocast(enabled=scaler is not None):
        #with torch.amp.autocast(enabled=scaler is not None):
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values()) #single value

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
