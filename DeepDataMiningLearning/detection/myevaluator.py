import torch
import torchvision
import datetime
import os
import time
import numpy as np
from DeepDataMiningLearning.detection import utils
#from DeepDataMiningLearning.detection.coco_eval import CocoEvaluator
#from DeepDataMiningLearning.detection.coco_utils import get_coco_api_from_dataset
#pip install pycocotools
from pycocotools.coco import COCO #https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask
import pycocotools.mask as mask_util
import copy
import io
from contextlib import redirect_stdout
from tqdm.auto import tqdm
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F

#https://github.com/pytorch/vision/blob/main/references/detection/coco_eval.py
class CocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        if not isinstance(iou_types, (list, tuple)):
            raise TypeError(f"This constructor expects iou_types of type list or tuple, instead  got {type(iou_types)}")
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys()))) #[139]
        self.img_ids.extend(img_ids) #[139]

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        if iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions): #predictions, key=image_id, val=dict
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"] #10,4 in xyxy format
            # Convert from xyxy to xywh format for COCO
            if len(boxes) > 0:
                # boxes is in xyxy format, convert to xywh
                boxes_xywh = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    boxes_xywh.append([x1, y1, width, height])
                boxes = boxes_xywh
            else:
                boxes = []
            
            scores = prediction["scores"].tolist() #list of 10
            labels = prediction["labels"].tolist()

            # Debug: Print box conversion for first few detections
            # if len(coco_results) < 5:
            #     print(f"Debug: Converting boxes for image {original_id}: {len(boxes)} detections")
            #     if len(boxes) > 0:
            #         box = boxes[0]
            #         print(f"Debug: Sample converted box (xywh): [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
            #         print(f"Debug: Box dimensions - width: {box[2]:.1f}, height: {box[3]:.1f}")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results #create a list of 10 dicts, each with "image_id", and one box (4)

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = utils.convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results

def merge(img_ids, eval_imgs):
    all_img_ids = utils.all_gather(img_ids)
    all_eval_imgs = utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(imgs):
    with redirect_stdout(io.StringIO()):
        imgs.evaluate()
    return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))


def get_coco_api_from_dataset(dataset):
    # FIXME: This is... awful?
    # for _ in range(10):
    #     if isinstance(dataset, torchvision.datasets.CocoDetection):
    #         break
    #     if isinstance(dataset, torch.utils.data.Subset):
    #         dataset = dataset.dataset
    # if isinstance(dataset, torchvision.datasets.CocoDetection):
    #     return dataset.coco
     return convert_to_coco_api(dataset)

def convert_to_coco_api2(ds):#mykittidetectiondataset
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    ds_len=len(ds)
    print("convert to coco api:")
    progress_bar = tqdm(range(ds_len))
    for img_idx in range(ds_len):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"]
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2] #img is CHW
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2] #[xmin, ymin, xmax, ymax] in torch to [xmin, ymin, width, height] in COCO
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
        progress_bar.update(1)
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    #print("convert_to_coco_api",dataset["categories"])
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds

def convert_to_coco_api(ds):#mykittidetectiondataset
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    ds_len=len(ds)
    print("convert to coco api:")
    progress_bar = tqdm(range(ds_len))
    for img_idx in range(ds_len):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx] #img is [3, 1280, 1920], 
        image_id = targets["image_id"] #68400
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2] #img is CHW
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone() #torch.Size([23, 4])
        bboxes[:, 2:] -= bboxes[:, :2] #[xmin, ymin, xmax, ymax] in torch to [xmin, ymin, width, height] in COCO
        bboxes = bboxes.tolist() #23 list of [536.0, 623.0, 51.0, 18.0]
        labels = targets["labels"].tolist() #torch.Size([23]) -> list 23 [1,1,1]
        areas = targets["area"].tolist() #torch.Size([23]) -> list 23 []
        iscrowd = targets["iscrowd"].tolist() #torch.Size([23]) -> list
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i] #int
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
        progress_bar.update(1)
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    #print("convert_to_coco_api",dataset["categories"])
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds

def convert_to_coco_api_new(ds):
    """
    Converts a dataset into a COCO-style API-compatible object.
    Handles both:
      - custom datasets returning dict targets
      - CocoDetection datasets returning list-of-dicts targets
    """
    coco_ds = COCO()
    ann_id = 1
    #dataset = {"images": [], "categories": [], "annotations": []}
    # Add COCO-required top-level fields
    dataset = {
        "info": {
            "description": "Converted COCO-style dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "custom script",
            "date_created": "2025-10-03",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }
    categories = set()

    print("Converting dataset to COCO format...")

    for idx in tqdm(range(len(ds)), desc="Building COCO dataset"):
        img, targets = ds[idx]
        #[3, 426, 640], list of 20 dicts ('image_id' =139 'bbox' =[493.1, 174.34, 20.29, 108.31])
        # --- case 1: CocoDetection returns list of annotations ---
        if isinstance(targets, list):
            if len(targets) == 0:
                continue  # skip empty images
            image_id = targets[0].get("image_id", idx)
            #height, width = img.shape[-2:]
            # Handle both Tensor [C,H,W] and PIL.Image
            if torch.is_tensor(img):
                height, width = img.shape[-2:]
            else:
                width, height = img.size
            img_dict = {"id": image_id, "height": height, "width": width}
            dataset["images"].append(img_dict)

            for t in targets:
                ann = {}
                ann["image_id"] = image_id
                bbox = t.get("bbox", [0, 0, 0, 0])
                ann["bbox"] = [float(x) for x in bbox]
                ann["category_id"] = int(t.get("category_id", 1))
                ann["area"] = float(t.get("area", bbox[2] * bbox[3]))
                ann["iscrowd"] = int(t.get("iscrowd", 0))
                ann["id"] = ann_id
                ann_id += 1

                dataset["annotations"].append(ann)
                categories.add(ann["category_id"])

        # --- case 2: custom dataset returns dict target ---
        elif isinstance(targets, dict):
            image_id = (
                targets["image_id"].item()
                if torch.is_tensor(targets["image_id"])
                else targets["image_id"]
            )
            #height, width = img.shape[-2:]
            if torch.is_tensor(img):
                height, width = img.shape[-2:]
            else:
                width, height = img.size
            img_dict = {"id": image_id, "height": height, "width": width}
            dataset["images"].append(img_dict)

            boxes = targets["boxes"].clone()
            boxes[:, 2:] -= boxes[:, :2]
            boxes = boxes.tolist()

            labels = targets["labels"].tolist()
            areas = targets.get("area", [b[2] * b[3] for b in boxes])
            iscrowd = targets.get(
                "iscrowd", torch.zeros(len(boxes), dtype=torch.int64)
            ).tolist()

            for i in range(len(boxes)):
                ann = {
                    "image_id": image_id,
                    "bbox": boxes[i],
                    "category_id": int(labels[i]),
                    "area": float(areas[i]),
                    "iscrowd": int(iscrowd[i]),
                    "id": ann_id,
                }
                dataset["annotations"].append(ann)
                categories.add(int(labels[i]))
                ann_id += 1

        else:
            raise TypeError(
                f"Unsupported target type: {type(targets)}. "
                "Expected dict or list of dicts."
            )

    # --- categories section ---
    dataset["categories"] = [{"id": int(i), "name": str(i)} for i in sorted(categories)]

    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds

@torch.inference_mode()
def simplemodelevaluate(model, data_loader, device):
    """
    Evaluate a detection model on a COCO-style dataset.

    This version supports both:
      (1) custom datasets returning a single dict target per image:
            target = {
                "boxes": Tensor[N, 4],        # (xmin, ymin, xmax, ymax) in pixel coords
                "labels": Tensor[N],          # category IDs (int)
                "image_id": Tensor[1] or int,
                "area": Tensor[N],            # optional
                "iscrowd": Tensor[N],         # optional
            }

      (2) torchvision.datasets.CocoDetection, which returns:
            (image, list[dict]) where each dict has:
                {"bbox": [x, y, w, h], "category_id": int, "image_id": int, ...}
          In this case, this function automatically extracts image_id = target[0]["image_id"]

    Args:
        model (torch.nn.Module): Trained detection model (e.g. FasterRCNN, DETR)
        data_loader (torch.utils.data.DataLoader): DataLoader returning (images, targets)
            where:
                - images: list[Tensor[C,H,W]] of images
                - targets: list[dict] with keys {"boxes", "labels", "image_id", ...}
        device (torch.device): CUDA or CPU device for inference

    Returns:
        coco_evaluator (CocoEvaluator): COCO evaluation object with accumulated metrics
    """

    # Device setup: evaluation runs on GPU if available
    device = torch.device(device)
    cpu_device = torch.device("cpu")

    model.eval().to(device)
    print(f"\n[INFO] Starting evaluation on device: {device}")

    ## Get COCO-style API object from dataset (it builds COCO-like ground-truth dict)
    coco = convert_to_coco_api_new(data_loader.dataset) #go through the whole dataset, convert_to_coco_api
    #coco = convert_to_coco_api(data_loader.dataset)
    #coco: pycocotools.coco.COCO object (ground truth)
    
    print("GT categories:", sorted({ann['category_id'] for ann in coco.dataset['annotations']})) #1-90
    
    # Evaluate only bounding boxes
    iou_types = ["bbox"] #_get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    # Optional timers for performance diagnostics
    model_time = 0.0
    evaluator_time = 0.0

    # tqdm progress bar for visualization
    progress_bar = tqdm(data_loader, desc="Evaluating", unit="batch")

    #for images, targets in data_loader: #images, targets are a tuple (tensor, )
    for images, targets in progress_bar:
        # new---------------------------------------------------------
        # images: tuple/list of images in this batch
        #   each image: Tensor[3, H, W], unnormalized, different sizes
        # targets: tuple/list of per-image annotations
        #   if using CocoDetection -> list[dict, dict, ...]
        #   if custom dataset     -> dict with "boxes", "labels", ...
        # ---------------------------------------------------------

        # images: list[Tensor[3,H,W]], typically batch of 1–4 images
        # targets: list[dict] with per-image ground truth
        #images = list(img.to(device) for img in images) #list of torch.Size([3, 426, 640]), len=1
        # Move inputs to GPU
        images = [img.to(device, non_blocking=True) for img in images]

        #targets: len=1 dict (image_id=139), boxes[20,4], labels[20]
        # Record inference time
        start_model_time = time.perf_counter()
        outputs = model(images)  # forward pass on GPU
        if device.type == "cuda":
            torch.cuda.synchronize()
        model_time += time.perf_counter() - start_model_time

        # ---------------------------------------------------------
        # outputs: list of dicts, same length as images
        #   each element:
        #     {
        #       "boxes": Tensor[M, 4]  (predicted boxes, xyxy)
        #       "labels": Tensor[M]    (predicted category IDs)
        #       "scores": Tensor[M]    (confidence)
        #     }
        # ---------------------------------------------------------

        # Move predictions to CPU for COCO evaluation (pycocotools requires numpy)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        #print("Sample prediction:", outputs[0]["boxes"][:2], outputs[0]["scores"][:2], outputs[0]["labels"][:2])
        #print("Pred categories sample:", outputs[0]['labels'][:10])
        # Prepare results mapping: image_id → predictions
        # Ensure image_id is converted to Python int (not tensor)
        #res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        # ---------------------------------------------------------
        # Prepare results mapping: image_id → model_output
        # ---------------------------------------------------------
        res = {}
        for target, output in zip(targets, outputs):
            # Case 1: COCO raw format (list of annotation dicts)
            if isinstance(target, list):
                if len(target) == 0:
                    # image has no ground-truth objects; skip
                    continue
                image_id = target[0].get("image_id", -1)
            # Case 2: Dict format from custom dataset
            elif isinstance(target, dict):
                image_id = (
                    target["image_id"].item()
                    if torch.is_tensor(target["image_id"])
                    else target["image_id"]
                )
            else:
                raise TypeError(
                    f"[ERROR] Unexpected target type {type(target)} "
                    f"— expected dict or list[dict]"
                )
        
            # Optional: filter out low-confidence predictions (<0.05)
            if "scores" in output and len(output["scores"]) > 0:
                keep = output["scores"] > 0.05
                if keep.sum() == 0:
                    continue  # all predictions below threshold
                output = {k: v[keep] for k, v in output.items()}
            else:
                continue  # no predictions at all

            # image_id must be an integer key
            res[int(image_id)] = output


        # Record evaluator time
        start_eval_time = time.perf_counter()
        if len(res) == 0:
            print("[Warning] No predictions in this batch!")
            continue
        else:
            for img_id, out in res.items():
                print(f"Debug: Image {img_id}, {len(out['boxes'])} boxes, avg score {out['scores'].mean().item() if len(out['scores'])>0 else 0:.3f}")
        coco_evaluator.update(res)
        evaluator_time += time.perf_counter() - start_eval_time

    # Final synchronization for distributed setups (optional)
    # coco_evaluator.synchronize_between_processes()

    # Accumulate results and summarize metrics
    print("\n[INFO] Accumulating COCO metrics...")
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # Print timing breakdown
    num_images = len(data_loader.dataset)
    print("\n[INFO] Evaluation complete.")
    print(f"  • Total images evaluated: {num_images}")
    print(f"  • Avg model (GPU) inference time per batch: {model_time / len(data_loader):.4f} s")
    print(f"  • Avg evaluator (CPU) update time per batch: {evaluator_time / len(data_loader):.4f} s\n")
    return coco_evaluator

@torch.inference_mode()
def modelevaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = utils._get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        #print("res:", res)
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

def yoloconvert_to_coco_api(ds):#mykittidetectiondataset
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    ds_len=len(ds)
    print("convert to coco api:")
    progress_bar = tqdm(range(ds_len))
    #for img_idx in range(ds_len):
    for batch in ds:
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        #img, targets = ds[img_idx]
        #batch = ds[img_idx]
        img = batch['img'] #[3, 640, 640] [1, 3, 640, 640]
        image_id = batch['image_id'][0] #targets["image_id"] 0
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2] #img is CHW
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = batch['bboxes'].clone() #normalized xc, yc, width, height
        #bboxes = targets["boxes"].clone()
        W=640
        H=640
        bboxes[:,0]=bboxes[:,0]*W  # xc * W
        bboxes[:,1]=bboxes[:,1]*H  # yc * H
        bboxes[:,2]=bboxes[:,2]*W  # width * W
        bboxes[:,3]=bboxes[:,3]*H  # height * H
        bboxes[:,0]=bboxes[:,0]-bboxes[:,2]/2 #xc - w/2 = xmin
        bboxes[:,1]=bboxes[:,1]-bboxes[:,3]/2 #yc - h/2 = ymin
        # Keep width and height as is for COCO format [xmin, ymin, width, height]
        # bboxes is now [xmin, ymin, width, height] which is correct for COCO
        bboxes = bboxes.tolist()
        labels = batch['cls'].tolist()
        #labels = targets["labels"].tolist()
        areas = batch["area"][0].tolist()
        #areas = targets["area"].tolist()
        iscrowd = batch["iscrowd"][0].tolist()
        #iscrowd = targets["iscrowd"].tolist()
        num_objs = len(bboxes)
        
        # Debug: Print ground truth info for first few images
        if len(dataset["images"]) <= 10:  # Only for first 10 images to avoid spam
            print(f"\n=== DEBUG: Processing image {image_id} (batch {len(dataset['images'])}) ===")
            print(f"Number of objects: {num_objs}")
            print(f"Ground truth labels: {labels}")
            print(f"Unique classes in GT: {set(labels)}")
            if len(bboxes) > 0:
                print(f"Ground truth bboxes (first 3): {bboxes[:3]}")
                print(f"Ground truth areas (first 3): {areas[:3]}")
                # Check bbox sizes
                for i, bbox in enumerate(bboxes[:3]):
                    width, height = bbox[2], bbox[3]
                    print(f"  GT bbox {i}: width={width:.1f}, height={height:.1f}")
            else:
                print("⚠️ NO GROUND TRUTH BBOXES FOUND!")
            print("=" * 50)
        
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            dataset["annotations"].append(ann)
            ann_id += 1
        
        # Debug: Print ground truth info for first few batches
        if len(dataset["images"]) <= 10:  # Only for first 10 images to avoid spam
            print(f"\n=== DEBUG: Processing image {image_id} (batch {len(dataset['images'])}) ===")
            print(f"Number of objects: {len(labels)}")
            print(f"Ground truth labels: {labels}")
            print(f"Unique classes in GT: {set(labels)}")
            if len(bboxes) > 0:
                print(f"Ground truth bboxes (first 3): {bboxes[:3]}")
                print(f"Ground truth areas (first 3): {areas[:3]}")
            else:
                print("⚠️ NO GROUND TRUTH BBOXES FOUND!")
            print("=" * 50)
        progress_bar.update(1)
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    #print("convert_to_coco_api",dataset["categories"])
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds

def yoloevaluate(model, data_loader, preprocess, device, use_coco_mapping=False):

    cpu_device = torch.device("cpu")
    model.eval()

    # Define COCO class mapping if requested
    coco_class_mapping = None
    if use_coco_mapping:
        # COCO class mapping - mapping from YOLO COCO classes to KITTI dataset classes
        # KITTI classes: Car(1), Van(2), Truck(3), Pedestrian(4), Person_sitting(5), Cyclist(6), Tram(7), Misc(8)
        # But KITTI validation set only has classes: 1, 3, 4, 6
        coco_class_mapping = {
            # COCO class ID -> KITTI class ID
            0: 4,   # person -> Pedestrian (KITTI class 4)
            1: 6,   # bicycle -> Cyclist (KITTI class 6)
            2: 1,   # car -> Car (KITTI class 1)
            3: 6,   # motorcycle -> Cyclist (KITTI class 6)
            5: 3,   # bus -> Truck (KITTI class 3)
            7: 3,   # train -> Truck (KITTI class 3)
            8: 3,   # truck -> Truck
            9: 3,   # boat -> Truck (closest match)
            # Additional mappings for commonly detected COCO classes
            11: 4,  # fire hydrant -> Pedestrian (treat as misc object)
            16: 4,  # cat -> Pedestrian (treat as misc object)
            20: 4,  # cow -> Pedestrian (treat as misc object)
            27: 4,  # handbag -> Pedestrian (treat as misc object)
            34: 4,  # kite -> Pedestrian (treat as misc object)
            38: 4,  # surfboard -> Pedestrian (treat as misc object)
            39: 4,  # tennis racket -> Pedestrian (treat as misc object)
            45: 4,  # spoon -> Pedestrian (treat as misc object)
            50: 4,  # orange -> Pedestrian (treat as misc object)
            55: 4,  # donut -> Pedestrian (treat as misc object)
            68: 4,  # cell phone -> Pedestrian (treat as misc object)
            71: 4,  # toaster -> Pedestrian (treat as misc object)
            73: 4,  # refrigerator -> Pedestrian (treat as misc object)
            # Note: Only keep predictions that map to valid KITTI classes
        }
        print(f"Using COCO to KITTI class mapping: {coco_class_mapping}")
        print("COCO->KITTI: person->Pedestrian, bicycle->Cyclist, car->Car, motorcycle->Cyclist, bus->Truck, truck->Truck")
        print(f"Valid KITTI classes in ground truth: [1, 3, 4, 6]")

    #coco = get_coco_api_from_dataset(data_loader.dataset) #go through the whole dataset, convert_to_coco_api
    #coco = yoloconvert_to_coco_api(data_loader)
    iou_types = ["bbox"] #_get_iou_types(model)
    #coco_evaluator = CocoEvaluator(coco, iou_types)
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {
        "images": [], 
        "categories": [], 
        "annotations": [],
        "info": {
            "description": "YOLO Dataset for Evaluation",
            "version": "1.0",
            "year": 2024,
            "contributor": "DeepDataMiningLearning",
            "date_created": "2024-01-01"
        }
    }
    categories = set()

    evalprogress_bar = tqdm(range(len(data_loader)))

    all_res=[]
    for batch in data_loader:
        targets={}
        #convert from yolo data format to COCO
        img = batch['img'] # [1, 3, 640, 640]
        img_dict = {}
        # Ensure image_id is an integer, not tensor
        image_id = batch['image_id'][0] if isinstance(batch['image_id'][0], int) else int(batch['image_id'][0])
        targets["image_id"]=image_id
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2] #img is CHW
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)

        box=batch['bboxes'] #normalized xc, yc, width, height
        box=torchvision.ops.box_convert(box, 'cxcywh', 'xyxy')#xcenter, ycenter,wh, to xyxy
        box=box.numpy()
        (H, W, C)=(640,640,3) #batch['orig_shape'][0] #tuple
        originalshape = [[H, W]]  # Fixed: scale_boxes expects (H,W) not (H,W,C)
        box[:,0]=box[:,0]*W
        box[:,1]=box[:,1]*H
        box[:,2]=box[:,2]*W
        box[:,3]=box[:,3]*H
        targets['boxes'] = box #xmin, ymin, xmax, ymax
        oneimg=torch.squeeze(img, 0) #[3, 640, 640]
        targets['labels']=batch['cls'].numpy()
        #vis_example(targets, oneimg, filename='result1.jpg')

        bboxes = batch['bboxes'].clone() #normalized xc, yc, width, height
        bboxes[:,0]=bboxes[:,0]*W
        bboxes[:,1]=bboxes[:,1]*H
        bboxes[:,2]=bboxes[:,2]*W
        bboxes[:,3]=bboxes[:,3]*H
        bboxes[:,0]=bboxes[:,0]-bboxes[:,2]/2 #-w/2: xmin
        bboxes[:,1]=bboxes[:,1]-bboxes[:,3]/2 #-H/2: ymin
        #[xmin, ymin, xmax, ymax] in torch to [xmin, ymin, width, height] in COCO
        bboxes = bboxes.tolist()
        labels = batch['cls'].tolist()
        #labels = targets["labels"].tolist()
        areas = batch["area"][0].tolist()
        #areas = targets["area"].tolist()
        iscrowd = batch["iscrowd"][0].tolist()
        #iscrowd = targets["iscrowd"].tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            dataset["annotations"].append(ann)
            ann_id += 1
    
        #Inference
        #img is already a tensor, preprocess function only do device
        imgtensors = batch['img'].to(device)
        
        # Clear cache to prevent memory buildup
        torch.cuda.empty_cache()

        #images = list(img.to(device) for img in images) #list of torch.Size([3, 426, 640]), len=1
        #targets: len=1 dict (image_id=139), boxes[20,4], labels[20]
        model_time = time.time()
        #outputs = model(images) #len1 dict boxes (10x4), labels[10], scores
        with torch.no_grad():  # Disable gradient computation to save memory
            preds = model(imgtensors)
        imgsize = imgtensors.shape[2:] #640, 640
        outputs = model.postprocess(preds, originalshape, imgsize)
        #outputs["boxes"] (xmin, ymin, xmax, ymax) format ["scores"] ["labels"]

        # Debug: Check what the model actually outputs
        print(f"Debug: Model output type: {type(outputs)}")
        if outputs:
            print(f"Debug: First output type: {type(outputs[0])}")
            if hasattr(outputs[0], 'boxes'):
                print(f"Debug: Output has boxes attribute")
                if outputs[0].boxes is not None:
                    print(f"Debug: Number of detections: {len(outputs[0].boxes)}")
                    if len(outputs[0].boxes) > 0:
                        print(f"Debug: Box tensor shape: {outputs[0].boxes.xyxy.shape}")
                        print(f"Debug: Confidence shape: {outputs[0].boxes.conf.shape}")
                        print(f"Debug: Class shape: {outputs[0].boxes.cls.shape}")
                        print(f"Debug: Sample confidences: {outputs[0].boxes.conf[:5].tolist()}")
                        print(f"Debug: Sample classes: {outputs[0].boxes.cls[:5].tolist()}")
                else:
                    print(f"Debug: Boxes is None")
            else:
                print(f"Debug: Output has no boxes attribute")
                print(f"Debug: Output attributes: {dir(outputs[0])}")
        
        # Convert UltralyticsResult to dictionary format for COCO evaluation
        converted_outputs = []
        for output in outputs:
            if hasattr(output, 'boxes') and output.boxes is not None:
                # Access the correct attributes from UltralyticsBoxes
                boxes = output.boxes.xyxy.cpu()  # [N, 4] in xyxy format
                scores = output.boxes.conf.cpu()  # [N]
                labels = output.boxes.cls.cpu()  # [N]
                
                print(f"Debug: Found {len(boxes)} detections")
                if len(boxes) > 0:
                    print(f"Debug: Score range: {scores.min():.3f} - {scores.max():.3f}")
                    print(f"Debug: Labels: {labels.unique().tolist()}")
                    print(f"Debug: Raw box sample: {boxes[0]} (before scaling)")
                    
                    # Check if boxes need scaling - YOLOv8 outputs normalized coordinates [0,1]
                    # but we need pixel coordinates for COCO evaluation
                    if boxes.max() <= 1.0:
                        print("Debug: Boxes appear to be normalized, scaling to image size")
                        # Scale boxes from normalized [0,1] to pixel coordinates using ACTUAL image dimensions
                        img_height, img_width = originalshape[0], originalshape[1]  # Use actual image dimensions
                        print(f"Debug: Using actual image dimensions: {img_width}x{img_height}")
                        boxes[:, [0, 2]] *= img_width   # x coordinates
                        boxes[:, [1, 3]] *= img_height  # y coordinates
                        print(f"Debug: Scaled box sample: {boxes[0]} (after scaling to actual dimensions)")
                    else:
                        print("Debug: Boxes appear to be in pixel coordinates already")
                
                converted_outputs.append({
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels
                })
            else:
                print(f"Debug: No detections found for image {image_id}")
                # Create empty tensors for consistency
                converted_outputs.append({
                    'boxes': torch.empty((0, 4)),
                    'scores': torch.empty(0),
                    'labels': torch.empty(0)
                })
        
        outputs = converted_outputs
        
        # Apply COCO class mapping if requested
        if use_coco_mapping and coco_class_mapping:
            for output in outputs:
                if 'labels' in output and len(output['labels']) > 0:
                    # Map COCO classes to KITTI classes
                    mapped_labels = []
                    mapped_boxes = []
                    mapped_scores = []
                    
                    labels_tensor = output['labels']
                    boxes_tensor = output['boxes']
                    scores_tensor = output['scores']
                    
                    for i in range(len(labels_tensor)):
                        coco_class = int(labels_tensor[i].item())
                        confidence = float(scores_tensor[i].item())
                        
                        # Apply confidence threshold - COCO evaluation typically uses 0.5 or higher
                        if confidence < 0.3:  # Lower threshold to see if we get any matches
                            continue
                            
                        if coco_class in coco_class_mapping:
                            kitti_class = coco_class_mapping[coco_class]
                            mapped_labels.append(kitti_class)
                            mapped_boxes.append(boxes_tensor[i])
                            mapped_scores.append(scores_tensor[i])
                    
                    # Update output with mapped classes
                    if mapped_labels:
                        output['labels'] = torch.tensor(mapped_labels, dtype=torch.long)
                        output['boxes'] = torch.stack(mapped_boxes)
                        output['scores'] = torch.tensor(mapped_scores)
                        print(f"Debug: Mapped {len(mapped_labels)} predictions from COCO to KITTI classes")
                        print(f"Debug: Mapped classes: {sorted(set(mapped_labels))}")
                        print(f"Debug: Mapped scores range: {min(mapped_scores):.3f} - {max(mapped_scores):.3f}")
                        # Print first few bounding boxes to check format
                        if len(mapped_boxes) > 0:
                            sample_box = mapped_boxes[0]
                            print(f"Debug: Sample bbox format: {sample_box} (should be [x1,y1,x2,y2])")
                            print(f"Debug: Sample bbox values: x1={sample_box[0]:.1f}, y1={sample_box[1]:.1f}, x2={sample_box[2]:.1f}, y2={sample_box[3]:.1f}")
                    else:
                        # No valid mappings found, create empty tensors
                        output['labels'] = torch.tensor([], dtype=torch.long)
                        output['boxes'] = torch.empty((0, 4))
                        output['scores'] = torch.tensor([])
                        print(f"Debug: No valid COCO->KITTI mappings found for this image")
                    
                    # No need to replace since we're already working with dictionary format
        else:
            # Convert UltralyticsResult objects to dictionary format for consistency
            for i, output in enumerate(outputs):
                if hasattr(output, 'boxes'):
                    outputs[i] = {
                        'boxes': output.boxes.xyxy,
                        'scores': output.boxes.conf,
                        'labels': output.boxes.cls
                    }

        #vis_example(outputs[0], oneimg, filename='result2.jpg')

        #outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        targets = [targets] #make it a list
        # Ensure image_id in outputs matches the ground truth
        for output in outputs:
            if 'image_id' not in output:
                output['image_id'] = image_id
        res = {target["image_id"]: output for target, output in zip(targets, outputs)} #dict, key=139, val=dict[boxes] 10,4
        
        # Debug: Print prediction details for first few batches
        if len(all_res) < 3:
            for img_id, pred in res.items():
                print(f"\nDebug: Image {img_id} predictions:")
                if isinstance(pred, dict):
                    if 'boxes' in pred and len(pred['boxes']) > 0:
                        print(f"  - {len(pred['boxes'])} boxes detected")
                        print(f"  - Score range: {pred['scores'].min():.3f} - {pred['scores'].max():.3f}")
                        print(f"  - Classes: {pred['labels'].unique().tolist()}")
                        # Print sample bounding box coordinates
                        sample_box = pred['boxes'][0]
                        print(f"  - Sample bbox: [{sample_box[0]:.1f}, {sample_box[1]:.1f}, {sample_box[2]:.1f}, {sample_box[3]:.1f}]")
                        print(f"  - Image dimensions: 640x640")
                    else:
                        print(f"  - No boxes in prediction dict")
                else:
                    print(f"  - Prediction type: {type(pred)}")
        
        evaluator_time = time.time()
        all_res.append(res)
        #coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        evalprogress_bar.update(1)

    #for coco evaluation
    if use_coco_mapping and coco_class_mapping:
        # When using COCO mapping, create categories based on KITTI classes
        kitti_class_names = {
            1: "Car", 2: "Van", 3: "Truck", 4: "Pedestrian", 
            5: "Person_sitting", 6: "Cyclist", 7: "Tram", 8: "Misc"
        }
        # Include ALL KITTI categories that appear in ground truth, not just mapped ones
        all_gt_categories = set()
        for res in all_res:
            for pred in res.values():
                if 'labels' in pred:
                    all_gt_categories.update(pred.get('original_labels', []))
        
        # Also include categories from the dataset
        all_gt_categories.update(categories)
        
        # Create categories for all classes that appear in either predictions or ground truth
        mapped_categories = set(coco_class_mapping.values())
        all_categories = mapped_categories.union(all_gt_categories)
        
        dataset["categories"] = [{"id": i, "name": kitti_class_names.get(i, f"class_{i}")} 
                               for i in sorted(all_categories)]
        print(f"Created COCO evaluation categories for KITTI classes: {[cat['name'] for cat in dataset['categories']]}")
        print(f"Ground truth categories: {sorted(categories)}")
        print(f"Mapped prediction categories: {sorted(mapped_categories)}")
    else:
        dataset["categories"] = [{"id": i, "name": str(i)} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()

    # Check if we have any valid predictions before evaluation
    if not all_res or all(not res for res in all_res):
        print("No valid predictions found for evaluation. Skipping COCO evaluation.")
        return

    # Debug: Count total predictions and check IoU potential
    total_predictions = 0
    predictions_by_class = {}
    gt_by_class = {}
    
    for res in all_res:
        for img_id, pred in res.items():
            if isinstance(pred, dict) and 'labels' in pred:
                total_predictions += len(pred['labels'])
                for label in pred['labels'].tolist():
                    predictions_by_class[label] = predictions_by_class.get(label, 0) + 1
    
    # Count ground truth by class
    for ann in dataset['annotations']:
        cat_id = ann['category_id']
        gt_by_class[cat_id] = gt_by_class.get(cat_id, 0) + 1
    
    print(f"\nDebug: Total predictions across all images: {total_predictions}")
    print(f"Debug: Total ground truth annotations: {len(dataset['annotations'])}")
    print(f"Debug: Ground truth categories: {[cat['id'] for cat in dataset['categories']]}")
    print(f"Debug: Predictions by class: {predictions_by_class}")
    print(f"Debug: Ground truth by class: {gt_by_class}")
    
    # Check for class overlap
    pred_classes = set(predictions_by_class.keys())
    gt_classes = set(gt_by_class.keys())
    overlap = pred_classes.intersection(gt_classes)
    print(f"Debug: Class overlap between predictions and GT: {overlap}")
    if not overlap:
        print("WARNING: No class overlap between predictions and ground truth!")

    # Create COCO evaluator with lower IoU thresholds to test if tiny boxes can match
    coco_evaluator = CocoEvaluator(coco_ds, iou_types)
    
    # Debug: Test with a very low IoU threshold to see if we get any matches
    print("Testing with standard COCO evaluation...")
    
    # Also test manual IoU calculation for a few samples
    if all_res:
        print("=== SPATIAL ANALYSIS ===")
        # Check multiple predictions and GT boxes from the same image
        for i, res_dict in enumerate(all_res[:3]):  # Check first 3 images
            for img_id, sample_res in res_dict.items():
                if isinstance(sample_res, dict) and 'boxes' in sample_res and len(sample_res['boxes']) > 0:
                    print(f"\nImage {img_id}:")
                    
                    # Show first few predictions
                    pred_boxes = sample_res['boxes'][:3]  # First 3 predictions
                    for j, pred_box in enumerate(pred_boxes):
                        print(f"  Prediction {j}: [{pred_box[0]:.1f}, {pred_box[1]:.1f}, {pred_box[2]:.1f}, {pred_box[3]:.1f}] (w={pred_box[2]-pred_box[0]:.1f}, h={pred_box[3]-pred_box[1]:.1f})")
                    
                    # Show ground truth boxes for the same image
                    gt_boxes = [ann for ann in dataset['annotations'] if ann['image_id'] == img_id]
                    print(f"  Ground truth boxes ({len(gt_boxes)} total):")
                    for j, gt_ann in enumerate(gt_boxes[:3]):  # First 3 GT boxes
                        gt_box = gt_ann['bbox']
                        print(f"    GT {j}: [{gt_box[0]:.1f}, {gt_box[1]:.1f}, {gt_box[2]:.1f}, {gt_box[3]:.1f}] (w={gt_box[2]:.1f}, h={gt_box[3]:.1f})")
                    
                    break  # Only check first image per batch
                if i >= 2:  # Only check first 3 images total
                    break
    
    for res in all_res:
        if res:  # Only update if res is not empty
            # Debug: Print prediction info for first few results
            if len([r for r in all_res if r]) <= 3:
                for img_id, pred in res.items():
                    if 'labels' in pred and len(pred['labels']) > 0:
                        print(f"Predictions for image {img_id}: {len(pred['labels'])} objects with classes {pred['labels'].tolist()}")
                        # Also print corresponding ground truth for comparison
                        gt_for_img = [ann for ann in dataset['annotations'] if ann['image_id'] == img_id]
                        gt_classes = [ann['category_id'] for ann in gt_for_img]
                        print(f"Ground truth for image {img_id}: {len(gt_for_img)} objects with classes {gt_classes}")
            coco_evaluator.update(res)
        

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    #torch.set_num_threads(n_threads)
    #return coco_evaluator

from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
def vis_example(onedetection, imgtensor, filename='result.jpg'):
    #labels = [names[i] for i in detections["labels"]] #classes[i]
    #img=im0.copy() #HWC (1080, 810, 3)
    #img_trans=im0[..., ::-1].transpose((2,0,1))  # BGR to RGB, HWC to CHW
    #imgtensor = torch.from_numpy(img_trans.copy()) #[3, 1080, 810]
    #pred_bbox_tensor=torchvision.ops.box_convert(torch.from_numpy(onedetection["boxes"]), 'xywh', 'xyxy')
    pred_bbox_tensor=torch.from_numpy(onedetection["boxes"])
    #pred_bbox_tensor=torch.from_numpy(onedetection["boxes"])
    print(pred_bbox_tensor)
    pred_labels = onedetection["labels"].astype(str).tolist()
    #img: Tensor of shape (C x H x W) and dtype uint8.
    #box: Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format.
    #labels: Optional[List[str]]
    imgtensor_uint=torchvision.transforms.functional.convert_image_dtype(imgtensor, torch.uint8)
    box = draw_bounding_boxes(imgtensor_uint, boxes=pred_bbox_tensor,
                            labels=pred_labels,
                            colors="red",
                            width=4, font_size=40)
    im = to_pil_image(box.detach())
    # save a image using extension
    im.save(filename)

def get_coco_val_dataset(root="/datasets/coco2017"):
    annFile = f"{root}/annotations/instances_val2017.json"
    imgDir = f"{root}/val2017"

    # Define a minimal transform to convert PIL to tensor
    def transform(img, target):
        img = F.to_tensor(img)
        return img, target

    dataset = CocoDetection(imgDir, annFile, transforms=transform)
    return dataset

def test_fasterrcnn_cocoevaluation(dataset_path="/mnt/e/Shared/Dataset/coco2017"):  # <-- change this path
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.transform import GeneralizedRCNNTransform
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load default pretrained model
    # num_classes = 91
    # model = fasterrcnn_resnet50_fpn(pretrained=True)
    # # COCO has 91 classes (including background)
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # model.to(device)
    # model.eval()

    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device).eval()

    # Load COCO val2017 dataset
    dataset = get_coco_val_dataset(dataset_path)
    #standarded CocoDetection, returns: (image, [annotation_dict_1, annotation_dict_2, ...])
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    # Run evaluation
    print("Running evaluation...")
    coco_evaluator = simplemodelevaluate(model, data_loader, device)

    print("Evaluation finished.")
    coco_evaluator.summarize()


from DeepDataMiningLearning.detection import utils
from DeepDataMiningLearning.detection.dataset import get_dataset
class args:
    data_path = '/data/Datasets/kitti/' #'/data/cmpe249-fa23/COCOoriginal/' # #'/data/cmpe249-fa23/WaymoCOCO/' #'/data/cmpe249-fa23/coco/'
    annotationfile = '/data/cmpe249-fa23/coco/train2017.txt'
    weights = None
    test_only = True
    backend = 'PIL' #tensor
    use_v2 = False
    dataset = 'yolo'#'coco'


def yolo_test():
    from DeepDataMiningLearning.detection.models import create_detectionmodel
    is_train =False
    is_val =True
    datasetname='kitti'#'coco' #'waymococo' #'yolo'
    dataset, num_classes=get_dataset(datasetname, is_train, is_val, args, output_format="yolo")
    print("train set len:", len(dataset))
    test_sampler = torch.utils.data.SequentialSampler(dataset) #RandomSampler(dataset)#torch.utils.data.SequentialSampler(dataset)
    new_collate_fn = utils.mycollate_fn #utils.mycollate_fn
    data_loader_test = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=test_sampler, num_workers=0, collate_fn=new_collate_fn
    )
    # for batch in data_loader_test:
    #     print(batch.keys()) #['img', 'bboxes', 'cls', 'batch_idx']
    #     break
    # #batch=next(iter(data_loader_test))
    #print(batch.keys())

    device='cuda:0'
    model, preprocess, classes = create_detectionmodel('/Developer/DeepDataMiningLearning/DeepDataMiningLearning/detection/modules/yolov8.yaml', num_classes=80, trainable_layers=0, ckpt_file='/data/cmpe249-fa23/modelzoo/yolov8n_statedicts.pt', fp16=False, device= device)
    print(f"Debug: Model classes: {classes}")
    print(f"Debug: Number of model classes: {len(classes) if classes else 'None'}")
    model.to(device)

    yoloevaluate(model, data_loader_test, preprocess, device, use_coco_mapping=True)
    #simplemodelevaluate(model, data_loader_test, device)

if __name__ == "__main__":
    test_fasterrcnn_cocoevaluation()