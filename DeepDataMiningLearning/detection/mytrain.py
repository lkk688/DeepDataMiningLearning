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
from DeepDataMiningLearning.detection.models import get_torchvision_detection_models, modify_fasterrcnnheader
from DeepDataMiningLearning.detection.myevaluator import simplemodelevaluate, modelevaluate
try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.") #pip install -q torchinfo


#Select the visible GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3" #"0,1"

MACHINENAME='HPC'
USE_AMP=True #AUTOMATIC MIXED PRECISION
if MACHINENAME=='HPC':
    os.environ['TORCH_HOME'] = '/data/cmpe249-fa23/torchhome/'
    DATAPATH='/data/cmpe249-fa23/torchvisiondata/'
elif MACHINENAME=='Container':
    os.environ['TORCH_HOME'] = '/data/torchhome/'
    DATAPATH='/data/torchvisiondata'
else:
    DATAPATH='./data'

#dataset: #coco, /data/cmpe249-fa23/COCOoriginal/
#kitti /data/cmpe249-fa23/torchvisiondata/Kitti/


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="/data/cmpe249-fa23/WaymoCOCO/", type=str, help="dataset path")
    parser.add_argument("--annotationfile", default=" annotations_train200new.json", type=str, help="dataset annotion file path, e.g., coco json file")
    parser.add_argument(
        "--dataset",
        default="waymococo", 
        type=str,
        help="dataset name. Use coco for object detection and instance segmentation and coco_kp for Keypoint detection",
    )
    parser.add_argument("--model", default="fasterrcnn_resnet50_fpn_v2", type=str, help="model name")
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
    parser.add_argument("--output-dir", default="/data/cmpe249-fa23/trainoutput", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint") #/data/cmpe249-fa23/trainoutput/kitti/model_4.pth
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=-1, type=int) #3
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument(
        "--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone"
    )
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
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")

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
        args.output_dir = os.path.join(args.output_dir, args.dataset)
        utils.mkdir(args.output_dir)

    #utils.init_distributed_mode(args)
    args.distributed = False
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset(args.dataset, is_train=True, is_val=False, args=args) #get_dataset
    dataset_test, _ = get_dataset(args.dataset, is_train=False, is_val=True, args=args)

    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    # idxsplit=int(len(indices)*0.80)#159948
    # dataset = torch.utils.data.Subset(dataset, indices[:idxsplit])
    # dataset_test = torch.utils.data.Subset(dataset, indices[idxsplit+1:])
    # dataset_test.transform = get_transform(is_train=False, args=args)
    print("train set len:", len(dataset))
    print("Test set len:", len(dataset_test))

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

    print("Creating model")
    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
    if args.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    
    model, preprocess, weights, classes = get_torchvision_detection_models(args.model)
    if len(classes) != num_classes:
        model = modify_fasterrcnnheader(model, num_classes, freeze=True)
    model.to(device)
    summary(model=model, 
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    ) 
    
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

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

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
        simplemodelevaluate(model, data_loader_test, device=device)
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
            modelevaluate(model, data_loader_test, device=device)

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
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

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
