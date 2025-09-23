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
from DeepDataMiningLearning.detection.models import create_detectionmodel #get_torchvision_detection_models, modify_fasterrcnnheader
from DeepDataMiningLearning.detection.modules.yolomodels import create_yolomodel
from DeepDataMiningLearning.detection.myevaluator import simplemodelevaluate, modelevaluate, yoloevaluate
from DeepDataMiningLearning.detection.ultralytics_converter import ensure_custom_yolo_checkpoint

# Ultralytics YOLO imports (optional)
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("[INFO] Ultralytics package not found. Install with: pip install ultralytics")

try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.") #pip install -q torchinfo


#Select the visible GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3" #"0,1"

#MACHINENAME='HPC'
USE_AMP=True #AUTOMATIC MIXED PRECISION
# if MACHINENAME=='HPC':
#     os.environ['TORCH_HOME'] = '/data/cmpe249-fa23/torchhome/'
#     DATAPATH='/data/cmpe249-fa23/torchvisiondata/'
# elif MACHINENAME=='Container':
#     os.environ['TORCH_HOME'] = '/data/torchhome/'
#     DATAPATH='/data/torchvisiondata'
# else:
#     DATAPATH='./data'

#dataset: #coco, /data/cmpe249-fa23/COCOoriginal/
#kitti /data/cmpe249-fa23/torchvisiondata/Kitti/

#(mycondapy310) [010796032@cs003 detection]$ torchrun --nproc_per_node=4 mytrain.py --batch-size=32
def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)
        # Ultralytics YOLO training parameters
    parser.add_argument("--use-ultralytics", default=False, action="store_true", help="Use Ultralytics YOLO package for training instead of custom torch implementation")
    parser.add_argument("--ultralytics-model", default="yolov8x.pt", type=str, help="Ultralytics YOLO model name (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)")
    parser.add_argument("--ultralytics-config", default=None, type=str, help="Path to Ultralytics YOLO config file (optional)")

    parser.add_argument("--data-path", default="/data/Datasets/kitti/", type=str, help="dataset path") #/data/cmpe249-fa23/waymotrain200cocoyolo/, /data/cmpe249-fa23/coco/
    parser.add_argument("--annotationfile", default="", type=str, help="dataset annotion file path, e.g., coco json file")
    parser.add_argument(
        "--dataset",
        default="kitti", 
        type=str,
        help="dataset name. Use coco for object detection, kitti for KITTI dataset, kitti_yolo for KITTI with YOLO format, yolo for YOLO dataset",
    )
    parser.add_argument("--model", default="yolov8", type=str, help="model name") #customrcnn_resnet152, fasterrcnn_resnet50_fpn_v2
    parser.add_argument("--scale", default="x", type=str, help="model scale: n, x") 
    parser.add_argument("--ckpt", default="", type=str, help="model name") #/data/cmpe249-fa23/modelzoo/yolov8x_statedicts.pt "/data/cmpe249-fa23/modelzoo/yolov8n_statedicts.pt"
    parser.add_argument("--trainable", default=0, type=int, help="number of trainable layers (sequence) of backbone")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=16, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--saveeveryepoch", default=1, type=int, metavar="N", help="number of epochs to save")
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
    parser.add_argument("--print-freq", default=50, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./output", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint") #/data/cmpe249-fa23/trainoutput/yolo/yolov8x0318/model_25.pth
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
        type=bool, 
        help="Only test the model",
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
    parser.add_argument("--expname", default="", help="experiment name, create a sub-folder")
    parser.add_argument("--img-size", "--imgsz", default=640, type=int, help="input image size for training and validation (default: 640)")
    

    return parser

def main(args):

    # Check if using Ultralytics YOLO
    if args.use_ultralytics:
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics package is required for --use-ultralytics option. Install with: pip install ultralytics")
        print("Using Ultralytics YOLO package for training")
        return train_with_ultralytics(args)
    
    # Support for multiple dataset types including KITTI
    supported_datasets = ["coco", "coco_kp", "kitti", "kitti_yolo", "yolo", "waymococo"]
    if args.dataset not in supported_datasets:
        raise ValueError(f"Dataset should be one of {supported_datasets}, got {args.dataset}")

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

    # Data loading code with optimized settings for different datasets
    print("Loading data")

    # Custom YOLO training path - use our dataset with YOLO format
    output_format = "yolo"
    print("🔧 Using custom YOLO training path with our KittiDataset")
    
    dataset, num_classes = get_dataset(args.dataset, is_train=True, is_val=False, args=args, output_format=output_format)
    dataset_test, _ = get_dataset(args.dataset, is_train=False, is_val=True, args=args, output_format=output_format)

    print(f"Dataset: {args.dataset}")
    print(f"Number of classes: {num_classes}")
    print("train set len:", len(dataset))
    print("Test set len:", len(dataset_test))

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # Optimize data loading - all datasets now support YOLO format
    if output_format == "yolo":
        # For YOLO format, use simpler batch sampling since data is already normalized
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
        print(f"Using optimized batch sampling for YOLO format")
    else:
        # Use aspect ratio grouping for torch/coco formats if enabled
        if args.aspect_ratio_group_factor >= 0:
            group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
            train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
            print(f"Using aspect ratio grouping for {output_format} format")
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    new_collate_fn = utils.mycollate_fn #utils.collate_fn
    
    # Optimize number of workers based on dataset type
    num_workers = args.workers
    if args.dataset in ["kitti", "kitti_yolo"] and num_workers > 2:
        # KITTI dataset is smaller, reduce workers to avoid overhead
        num_workers = min(2, args.workers)
        print(f"Optimized num_workers to {num_workers} for {args.dataset} dataset")
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=num_workers, collate_fn=new_collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=1, collate_fn=new_collate_fn
    )

    print("Creating model")
    #kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
    kwargs = {"trainable": args.trainable}
    if args.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    
    # Handle checkpoint conversion for custom YOLO models when using custom training
    if args.model.startswith('yolo'):
        # If no checkpoint specified but we're using custom YOLO, try to find/convert one
        if not args.ckpt:
            print(f"No checkpoint specified for {args.model}. Attempting to find or convert from Ultralytics...")
            
            # Try to ensure custom checkpoint exists
            converted_ckpt = ensure_custom_yolo_checkpoint(
                model_name=args.model+args.scale,
                ckpt_path=None,  # Let the function determine the path
                weights_dir=os.path.join(args.output_dir, "weights"), #"./weights",
                ckpt_dir= os.path.join(args.output_dir, "checkpoints"), # "./checkpoints",
                num_classes=num_classes,
                device=device
            )
            
            if converted_ckpt:
                print(f"Using converted checkpoint: {converted_ckpt}")
                args.ckpt = converted_ckpt
            else:
                print("No checkpoint available. Training will start from scratch.")
        
        # If checkpoint is specified but doesn't exist, try to convert
        elif not os.path.exists(args.ckpt):
            print(f"Checkpoint not found: {args.ckpt}")
            print("Attempting to convert from Ultralytics model...")
            
            # Try to ensure custom checkpoint exists
            converted_ckpt = ensure_custom_yolo_checkpoint(
                model_name=args.model,
                ckpt_path=args.ckpt,
                weights_dir="./weights",
                ckpt_dir=os.path.dirname(args.ckpt) if os.path.dirname(args.ckpt) else "./checkpoints",
                num_classes=num_classes,
                device=device
            )
            
            if converted_ckpt:
                print(f"Successfully converted checkpoint: {converted_ckpt}")
                args.ckpt = converted_ckpt
            else:
                print("Failed to convert checkpoint. Training will start from scratch.")
                args.ckpt = None
    
    # Enhanced model creation with better KITTI support
    #model, preprocess, classes = create_detectionmodel(args.model, num_classes, args.trainable, ckpt_file=args.ckpt, fp16=False, device= device, scale=args.scale)
    # Create YOLOv11 model
    model, preprocess, classes = create_yolomodel(
        modelname=args.model, #'yolov11n', 'yolov12n'
        num_classes=num_classes, #80,
        ckpt_file=args.ckpt,
        device=device, 
        scale=args.scale
    )

    # Print model information for better debugging
    print(f"Model: {args.model}")
    print(f"Number of classes: {num_classes}")#8
    print(f"Model classes: {classes}")
    
    #model.to(device)
    
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.norm_weight_decay is None:
        parameters = [p for p in model.parameters() if p.requires_grad]#add parameters to list
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    opt_name = args.opt.lower() #'sgd'
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

    scaler = torch.cuda.amp.GradScaler() if args.amp else None #amp False

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
        print(f"Running evaluation on {args.dataset} dataset...")
        
        # Use appropriate evaluator based on dataset type
        if args.dataset in ["kitti", "kitti_yolo"]:
            print("Using YOLO evaluator for KITTI dataset")
            yoloevaluate(model, data_loader_test, preprocess, device)
        else:
            # For other datasets, use the standard YOLO evaluator
            yoloevaluate(model, data_loader_test, preprocess, device)
        
        return

    print("Start training")
    start_time = time.time()
    
    # Enhanced training loop with better logging and KITTI support
    for epoch in range(args.start_epoch, args.epochs+1):
        if args.distributed:
            train_sampler.set_epoch(epoch) #set the epoch for the train sampler
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Dataset: {args.dataset}, Batch size: {args.batch_size}")
        
        train_one_epoch(model, optimizer, data_loader, device, preprocess, epoch, args.print_freq, scaler)
        lr_scheduler.step()
        
        # Periodic evaluation during training for KITTI
        if args.dataset in ["kitti", "kitti_yolo"] and epoch % 5 == 0:
            print(f"Running periodic evaluation at epoch {epoch}...")
            try:
                yoloevaluate(model, data_loader_test, preprocess, device)
            except Exception as e:
                print(f"Evaluation failed: {e}")
        
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
            yoloevaluate(model, data_loader_test, preprocess, device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

def train_one_epoch(model, optimizer, data_loader, device, preprocess, epoch, print_freq, scaler=None):
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

    #images, targets
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        batch['img']=preprocess(batch['img']) #batch['img'] = batch['img'].to(device)
        #img is already a tensor, preprocess function only do device

        #images = list(image.to(device) for image in images) #list of [3, 1280, 1920]
        #targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets] #tuple to list
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            #loss_dict = model(images, targets) #dict with 4 keys
            loss, loss_items = model(batch)
            losses = loss #sum(loss for loss in loss_dict.values()) #single value
            loss_dict={}
            loss_dict['box']=loss_items[0]
            loss_dict['cls']=loss_items[1]
            loss_dict['dfl']=loss_items[2]

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

def train_with_ultralytics(args):
    """
    Train YOLO model using Ultralytics package with data conversion
    """
    print(f"🚀 Training with Ultralytics YOLO: {args.ultralytics_model}")
    
    # Import data converter
    from DeepDataMiningLearning.detection.data_converter import export_kitti_for_ultralytics
    
    # Create output directory
    if args.output_dir:
        utils.mkdir(args.output_dir)
        if args.expname:
            output_dir = os.path.join(args.output_dir, "ultralytics", args.expname)
        else:
            output_dir = os.path.join(args.output_dir, "ultralytics")
        utils.mkdir(output_dir)
    else:
        output_dir = "./runs/detect/train"
    
    # Step 1: Export KITTI data to Ultralytics format (skip if already exists)
    data_export_dir = os.path.join(output_dir, "exported_data")
    yaml_config_path = os.path.join(data_export_dir, "dataset.yaml")
    
    if os.path.exists(yaml_config_path) and os.path.exists(data_export_dir):
        print(f"\n✅ Step 1: Data already converted, using existing format at: {data_export_dir}")
        print(f"   YAML config: {yaml_config_path}")
    else:
        print("\n📦 Step 1: Converting KITTI data to Ultralytics format...")
        yaml_config_path = export_kitti_for_ultralytics(args, data_export_dir)
    
    # Step 2: Initialize YOLO model
    print(f"\n🤖 Step 2: Initializing YOLO model: {args.ultralytics_model}")
    model = YOLO(args.ultralytics_model)
    
    # Convert device format for Ultralytics compatibility
    if args.device == 'cuda':
        # Check if CUDA is available, otherwise use CPU
        if torch.cuda.is_available():
            ultralytics_device = 0  # Use first GPU
        else:
            ultralytics_device = 'cpu'
            print("[WARNING] CUDA not available, using CPU for training")
    elif args.device == 'cpu':
        ultralytics_device = 'cpu'
    else:
        # If device is already in specific format (e.g., 'cuda:0', '0'), use as is
        ultralytics_device = args.device

    # Step 3: Set up training parameters
    train_params = {
        'data': yaml_config_path,  # Use exported data configuration
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': 640,  # Default image size for YOLO
        'lr0': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'project': output_dir,
        'name': f'exp_{args.expname}' if args.expname else 'exp',
        'device': ultralytics_device,
        'workers': args.workers,
        'save_period': args.saveeveryepoch if hasattr(args, 'saveeveryepoch') else -1,
        'patience': 50,  # Early stopping patience
        'save': True,
        'plots': True,
    }
    
    # Add config file if provided
    if hasattr(args, 'ultralytics_config') and args.ultralytics_config:
        train_params['cfg'] = args.ultralytics_config
    
    print(f"\n⚙️  Step 3: Training parameters:")
    for key, value in train_params.items():
        print(f"   {key}: {value}")
    
    # Step 4: Start training
    print(f"\n🏋️  Step 4: Starting Ultralytics training...")
    results = model.train(**train_params)
    
    # Step 5: Evaluation
    if not args.test_only:
        print("\n📊 Step 5: Starting evaluation...")
        metrics = model.val()
        print(f"✅ Training completed! Results saved to: {output_dir}")
        return results, metrics
    
    return results


def prepare_ultralytics_data_config(args):
    """
    Prepare data configuration for Ultralytics YOLO training
    """
    import yaml
    
    # Create a temporary data config file for Ultralytics
    config_path = os.path.join(args.output_dir or "./", "ultralytics_data.yaml")
    
    # Map dataset types to paths and classes
    if args.dataset == "kitti" or args.dataset == "kitti_yolo":
        # KITTI dataset configuration
        config = {
            'path': args.data_path,
            'train': 'training/image_2',  # KITTI training images
            'val': 'training/image_2',    # KITTI validation images (same as train for now)
            'test': 'testing/image_2',    # KITTI test images
            'nc': 10,  # Number of classes in KITTI
            'names': ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 
                     'Cyclist', 'Tram', 'Misc', 'DontCare', 'Background']
        }
    elif args.dataset == "yolo":
        # Generic YOLO dataset
        config = {
            'path': args.data_path,
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 80,  # Default COCO classes
            'names': list(range(80))  # Will be updated based on actual dataset
        }
    else:
        raise ValueError(f"Dataset {args.dataset} not supported for Ultralytics training")
    
    # Write config to file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created Ultralytics data config: {config_path}")
    return config_path


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
