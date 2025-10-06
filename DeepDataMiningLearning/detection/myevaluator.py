import math
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

import os, math, time
from tqdm import tqdm
import torch
import matplotlib
matplotlib.use("Agg")     # headless backend for PNG/PDF saving
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
import pandas as pd

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
    Handles:
      • custom datasets returning dict targets
      • CocoDetection datasets returning list-of-dict targets

    Returns:
        coco_ds (COCO): pycocotools-style COCO object.
    """
    import torch
    from pycocotools.coco import COCO
    from tqdm import tqdm

    coco_ds = COCO()
    ann_id = 1
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

        # --- Case 1: CocoDetection returns list of dicts ---
        if isinstance(targets, list):
            if len(targets) == 0:
                continue
            image_id = targets[0].get("image_id", idx)
            if torch.is_tensor(img):
                height, width = img.shape[-2:]
            else:
                width, height = img.size

            dataset["images"].append({"id": int(image_id), "height": int(height), "width": int(width)})

            for t in targets:
                bbox = t.get("bbox", [0, 0, 0, 0])
                if torch.is_tensor(bbox):
                    bbox = bbox.tolist()
                if len(bbox) != 4:
                    continue  # skip malformed
                w, h = bbox[2], bbox[3]
                if w <= 0 or h <= 0:
                    continue
                ann = {
                    "image_id": int(image_id),
                    "bbox": [float(x) for x in bbox],
                    "category_id": int(t.get("category_id", 1)),
                    "area": float(t.get("area", w * h)),
                    "iscrowd": int(t.get("iscrowd", 0)),
                    "id": ann_id,
                }
                dataset["annotations"].append(ann)
                categories.add(int(ann["category_id"]))
                ann_id += 1

        # --- Case 2: custom dataset returns dict target ---
        elif isinstance(targets, dict):
            # Safe image_id extraction
            image_id = int(targets["image_id"].item() if torch.is_tensor(targets["image_id"]) else targets["image_id"])
            if torch.is_tensor(img):
                height, width = img.shape[-2:]
            else:
                width, height = img.size

            dataset["images"].append({"id": image_id, "height": int(height), "width": int(width)})

            if "boxes" not in targets or len(targets["boxes"]) == 0:
                continue

            bboxes = targets["boxes"]
            if not torch.is_tensor(bboxes):
                bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
            if bboxes.ndim == 1:
                bboxes = bboxes.unsqueeze(0)
            bboxes = bboxes.clone().to(torch.float32)

            # ---- Fix flipped coordinates (x2<x1, y2<y1) ----
            x1 = torch.minimum(bboxes[:, 0], bboxes[:, 2])
            y1 = torch.minimum(bboxes[:, 1], bboxes[:, 3])
            x2 = torch.maximum(bboxes[:, 0], bboxes[:, 2])
            y2 = torch.maximum(bboxes[:, 1], bboxes[:, 3])
            bboxes = torch.stack([x1, y1, x2, y2], dim=1)

            # ---- Convert xyxy → xywh ----
            bboxes[:, 2:] -= bboxes[:, :2]  # width/height = x2-x1, y2-y1

            # ---- Remove invalid / zero-size ----
            valid = (bboxes[:, 2] > 0) & (bboxes[:, 3] > 0) & torch.isfinite(bboxes).all(dim=1)
            if not valid.all():
                bboxes = bboxes[valid]

            if bboxes.numel() == 0:
                continue

            # ---- Align labels, areas, iscrowd ----
            labels = targets.get("labels", torch.ones((len(bboxes),), dtype=torch.int64))
            if torch.is_tensor(labels):
                labels = labels.tolist()
            if len(labels) < len(bboxes):
                labels = (labels + [1] * len(bboxes))[: len(bboxes)]

            if "area" in targets and isinstance(targets["area"], torch.Tensor):
                areas = targets["area"].tolist()
            else:
                areas = (bboxes[:, 2] * bboxes[:, 3]).tolist()

            if "iscrowd" in targets and isinstance(targets["iscrowd"], torch.Tensor):
                iscrowd = targets["iscrowd"].tolist()
            else:
                iscrowd = [0] * len(bboxes)

            # ---- Create per-instance annotations ----
            for i in range(len(bboxes)):
                bbox = bboxes[i].tolist()
                w, h = bbox[2], bbox[3]
                if w <= 0 or h <= 0 or not np.isfinite(w) or not np.isfinite(h):
                    continue
                ann = {
                    "image_id": image_id,
                    "bbox": [float(v) for v in bbox],
                    "category_id": int(labels[i]),
                    "area": float(areas[i]),
                    "iscrowd": int(iscrowd[i]),
                    "id": ann_id,
                }
                dataset["annotations"].append(ann)
                categories.add(int(labels[i]))
                ann_id += 1

        else:
            raise TypeError(f"Unsupported target type: {type(targets)}. Expected dict or list of dicts.")

    # --- Build category section ---
    dataset["categories"] = [{"id": int(cid), "name": str(cid)} for cid in sorted(categories)]

    # --- Finalize COCO object ---
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    print(f"[INFO] Finished COCO conversion: {len(dataset['images'])} images, "
          f"{len(dataset['annotations'])} annotations, {len(categories)} categories.")
    return coco_ds

@torch.inference_mode()
def simplemodelevaluate_old(
    model,
    data_loader,
    device,
    class_map=None,   # Optional dict, e.g. {1:2, 3:1, ...}
    score_thresh=0.05
):
    """
    Evaluate a detection model on a COCO-style dataset.
    Optionally remaps predicted class IDs (e.g., COCO→Waymo).

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
        class_map (dict[int,int] or None): optional mapping of predicted class IDs,
            e.g. {1:2, 3:1, 10:4} for COCO→Waymo
        score_thresh (float): minimum confidence to keep a detection

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
        
        for i, out in enumerate(outputs):
            n_boxes = len(out["boxes"])
            avg_score = out["scores"].mean().item() if n_boxes > 0 else 0
            print(f"[DEBUG] Batch sample {i}: {n_boxes} boxes, avg score {avg_score:.4f}")
    
        # ------------------------
        # Optional class remapping
        # ------------------------
        if class_map is not None:
            for out in outputs:
                labels = out["labels"]
                mapped = torch.zeros_like(labels)
                for src, dst in class_map.items():
                    mapped[labels == src] = dst
                out["labels"] = mapped

        # ------------------------
        # Filter low-confidence predictions
        # ------------------------
        for out in outputs:
            if "scores" in out and len(out["scores"]) > 0:
                keep = out["scores"] > score_thresh
                out.update({k: v[keep] for k, v in out.items()})
            else:
                out["boxes"] = torch.zeros((0, 4))
                out["labels"] = torch.zeros((0,), dtype=torch.int64)
                out["scores"] = torch.zeros((0,))

        #print("Sample prediction:", outputs[0]["boxes"][:2], outputs[0]["scores"][:2], outputs[0]["labels"][:2])
        #print("Pred categories sample:", outputs[0]['labels'][:10])
        # Prepare results mapping: image_id → predictions
        # Ensure image_id is converted to Python int (not tensor)
        #res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        # ---------------------------------------------------------
        # Prepare results mapping: image_id → model_output
        # ---------------------------------------------------------
        # ------------------------
        # Build results dict
        # ------------------------
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
        
        # check every output has required keys and nonempty tensors
        valid_res = True
        for r in res.values():
            if "boxes" not in r or len(r["boxes"]) == 0:
                valid_res = False
        if not valid_res:
            print("[Warning] Skipping batch with invalid predictions.")
            continue
        
        # Sanity check predictions
        for img_id, out in res.items():
            if len(out["boxes"]) == 0:
                continue
            boxes = out["boxes"].numpy().tolist()
            for box in boxes:
                if len(box) != 4 or any(math.isnan(x) for x in box):
                    print(f"[Warning] Invalid prediction bbox for image {img_id}: {box}")
                    
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



def _evaluate_single_image(pred_boxes, gt_boxes, iou_thresh=0.5):
    """
    Compute basic detection statistics (TP, FP, FN) for a single image.

    This function performs a simple greedy IoU-based matching between
    predicted boxes and ground-truth boxes. It does NOT consider class labels
    or confidence scores — it only checks geometric overlap.

    Args:
        pred_boxes (Tensor[N, 4]): Predicted boxes in [x1, y1, x2, y2] format.
        gt_boxes (Tensor[M, 4]):   Ground-truth boxes in [x1, y1, x2, y2] format.
        iou_thresh (float):        IoU threshold for a detection to count as TP.

    Returns:
        dict with:
            - tp (int): number of true positives
            - fp (int): number of false positives
            - fn (int): number of false negatives
    """

    # --- Handle simple edge cases first ---
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        # No GT and no predictions → nothing to evaluate
        return dict(tp=0, fp=0, fn=0)
    if len(gt_boxes) == 0:
        # No GT but model predicted something → all are false positives
        return dict(tp=0, fp=len(pred_boxes), fn=0)
    if len(pred_boxes) == 0:
        # There are GT objects but model predicted nothing → all are false negatives
        return dict(tp=0, fp=0, fn=len(gt_boxes))

    # --- Compute IoU between every predicted and ground-truth box ---
    # IoU matrix has shape [N_pred, N_gt]
    iou_matrix = torchvision.ops.box_iou(pred_boxes, gt_boxes)

    # Track which GT boxes have already been matched to a prediction
    matched_gt = set()
    tp = 0

    # --- Greedy matching: for each prediction, find its best GT ---
    for i in range(len(pred_boxes)):
        # find GT box with max IoU for this prediction
        max_iou, j = iou_matrix[i].max(0)
        # Count as TP if IoU passes threshold and GT not matched yet
        if max_iou >= iou_thresh and j.item() not in matched_gt:
            tp += 1
            matched_gt.add(j.item())

    # --- Remaining unmatched predictions and GT boxes ---
    fp = len(pred_boxes) - tp              # false positives
    fn = len(gt_boxes) - tp                # false negatives

    return dict(tp=tp, fp=fp, fn=fn)

def xyxy_to_xywh_safe(boxes, img_w=None, img_h=None, verbose=True):
    """
    Convert [x1, y1, x2, y2] → [x, y, w, h] safely.

    Handles:
        • Swapped coordinates (x2 < x1 or y2 < y1)
        • Negative or zero width/height
        • NaN / Inf values
        • Clamping to image size if provided
        • Non-tensor / wrong-shape inputs

    Args:
        boxes (Tensor[N,4]): input boxes in [x1, y1, x2, y2].
        img_w, img_h (int or None): optional image size for clamping.
        verbose (bool): if True, prints warnings when bad boxes are found.

    Returns:
        Tensor[N,4]: valid [x, y, w, h] boxes (float32).
    """
    import torch

    # ---- 0️⃣ Handle empty inputs or wrong shapes ----
    if boxes is None or len(boxes) == 0:
        if verbose:
            print("[WARN] xyxy_to_xywh_safe: empty boxes input.")
        return torch.zeros((0, 4), dtype=torch.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"[ERROR] xyxy_to_xywh_safe: expected shape [N,4], got {boxes.shape}")

    boxes = boxes.clone().to(torch.float32)

    # ---- 1️⃣ Detect and fix flipped coordinates ----
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    flipped_x = (x2 < x1).sum().item()
    flipped_y = (y2 < y1).sum().item()
    if flipped_x > 0 or flipped_y > 0:
        if verbose:
            print(f"[WARN] xyxy_to_xywh_safe: {flipped_x} boxes had x2<x1, {flipped_y} boxes had y2<y1; auto-correcting.")
    x1_fixed = torch.minimum(x1, x2)
    y1_fixed = torch.minimum(y1, y2)
    x2_fixed = torch.maximum(x1, x2)
    y2_fixed = torch.maximum(y1, y2)

    # ---- 2️⃣ Compute width and height ----
    w = (x2_fixed - x1_fixed)
    h = (y2_fixed - y1_fixed)

    # ---- 3️⃣ Handle invalid (non-finite or non-positive) sizes ----
    nan_mask = ~torch.isfinite(torch.stack([x1_fixed, y1_fixed, w, h], dim=1))
    bad_mask = (w <= 0) | (h <= 0) | nan_mask.any(dim=1)
    n_bad = bad_mask.sum().item()
    if n_bad > 0:
        if verbose:
            print(f"[WARN] xyxy_to_xywh_safe: {n_bad} invalid boxes removed (non-finite or non-positive).")
        keep_mask = ~bad_mask
        x1_fixed, y1_fixed, w, h = x1_fixed[keep_mask], y1_fixed[keep_mask], w[keep_mask], h[keep_mask]

    boxes_xywh = torch.stack([x1_fixed, y1_fixed, w.clamp(min=1e-3), h.clamp(min=1e-3)], dim=1)

    # ---- 4️⃣ Clamp to image bounds if size provided ----
    if img_w is not None and img_h is not None:
        # Clamp x/y within image bounds
        boxes_xywh[:, 0].clamp_(min=0, max=img_w - 1)
        boxes_xywh[:, 1].clamp_(min=0, max=img_h - 1)

        # Limit width and height so that x + w ≤ img_w and y + h ≤ img_h
        boxes_xywh[:, 2].clamp_max_(img_w - boxes_xywh[:, 0])
        boxes_xywh[:, 3].clamp_max_(img_h - boxes_xywh[:, 1])

    # ---- 5️⃣ Final validation ----
    if (boxes_xywh[:, 2] <= 0).any() or (boxes_xywh[:, 3] <= 0).any():
        raise ValueError("[ERROR] xyxy_to_xywh_safe: negative/zero width or height after correction.")
    if not torch.isfinite(boxes_xywh).all():
        raise ValueError("[ERROR] xyxy_to_xywh_safe: NaN/Inf detected after correction.")

    return boxes_xywh

@torch.inference_mode()
def simplemodelevaluate(
    model,
    data_loader,
    device,
    class_map=None,
    class_names=None,
    score_thresh=0.05,
    vis_dir="output/debug_vis",
    max_vis=100,
    DEBUG=True,
):
    """
    Evaluate an object detection model (e.g., FasterRCNN, DETR) on a COCO-style dataset.

    Key features:
      • Works directly with COCOeval (expects model outputs in xyxy)
      • Collects all detections across dataset
      • Keeps optional per-image visualizations (before/after filtering)
      • Supports optional class ID remapping
    """
    import os, time, torch, numpy as np
    from tqdm import tqdm

    os.makedirs(vis_dir, exist_ok=True)
    device, cpu_device = torch.device(device), torch.device("cpu")
    model.eval().to(device)
    print(f"\n[INFO] Starting evaluation on device: {device}")

    # ------------------------------------------------------------------
    # Build COCO ground-truth API from dataset
    # ------------------------------------------------------------------
    coco_gt = convert_to_coco_api_new(data_loader.dataset)
    print("GT categories:", sorted({ann["category_id"] for ann in coco_gt.dataset["annotations"]}))
    coco_evaluator = CocoEvaluator(coco_gt, ["bbox"])

    model_time = evaluator_time = 0.0
    vis_count = 0
    progress_bar = tqdm(data_loader, desc="Evaluating", unit="batch")

    # Store results for all images
    res, img_map, target_map = {}, {}, {}

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    for images, targets in progress_bar:
        images = [img.to(device, non_blocking=True) for img in images]
        t0 = time.perf_counter()
        outputs = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        model_time += time.perf_counter() - t0

        outputs = [{k: v.to(cpu_device) for k, v in o.items()} for o in outputs]

        # --- Optional class remapping ---
        if class_map is not None:
            for out in outputs:
                mapped = torch.zeros_like(out["labels"])
                for src, dst in class_map.items():
                    mapped[out["labels"] == src] = dst
                out["labels"] = mapped

        # --- Process per-image ---
        for img, target, output in zip(images, targets, outputs):
            image_id = int(target["image_id"]) if isinstance(target, dict) else int(target[0]["image_id"])
            gt_boxes = target["boxes"].cpu()
            n_gt, n_before = len(gt_boxes), len(output["boxes"])

            # Filter low-confidence predictions
            keep = output["scores"] > score_thresh if "scores" in output else torch.zeros(0, dtype=torch.bool)
            output = {k: v[keep] for k, v in output.items()}
            n_after = len(output["boxes"])

            if DEBUG:
                print(f"[INFO] Image {image_id}: GT={n_gt}, predicted(before)={n_before}, after filter={n_after}")

            # Quick per-image TP/FP/FN diagnostic
            if DEBUG:
                stats = _evaluate_single_image(output["boxes"], gt_boxes)
                print(f"        TP={stats['tp']}, FP={stats['fp']}, FN={stats['fn']}")

            res[image_id] = output
            img_map[image_id] = img.cpu()
            target_map[image_id] = target

            # Visualization (raw xyxy)
            if vis_count < max_vis:
                vis_count += 1
                _visualize_prediction(
                    img.cpu(),
                    target,
                    output,
                    save_path=os.path.join(vis_dir, f"{image_id}_xyxy.jpg"),
                    class_names=class_names,
                    gt_box_type="xyxy",
                    pred_box_type="xyxy",
                )

    # ------------------------------------------------------------------
    # Convert tensors → numpy for COCOeval
    # ------------------------------------------------------------------
    for img_id, out in res.items():
        out["boxes"] = out["boxes"].cpu().numpy().astype(np.float32)
        out["scores"] = out["scores"].cpu().numpy().astype(np.float32)
        out["labels"] = out["labels"].cpu().numpy().astype(np.int32)
        res[img_id] = out

    print(f"[INFO] Prepared predictions for {len(res)} images")

    # ------------------------------------------------------------------
    # Optional debug info
    # ------------------------------------------------------------------
    if DEBUG:
        gt_ids = sorted({ann["category_id"] for ann in coco_evaluator.coco_gt.dataset["annotations"]})
        dt_ids = sorted({int(l) for r in res.values() for l in r["labels"].tolist()})
        print("GT category ids:", gt_ids)
        print("DT category ids:", dt_ids)

        sample_id = list(res.keys())[0]
        gt_labels = [ann["category_id"] for ann in coco_evaluator.coco_gt.dataset["annotations"]
                     if ann["image_id"] == sample_id]
        dt_labels = res[sample_id]["labels"]
        print(f"Sample GT labels ({sample_id}):", gt_labels[:10])
        print(f"Sample DT labels ({sample_id}):", dt_labels[:10])
        print("Sample boxes:", res[sample_id]["boxes"][:3])
        print("Sample scores:", res[sample_id]["scores"][:3])

    # ------------------------------------------------------------------
    # COCOeval expects xyxy input and performs its own xywh conversion
    # ------------------------------------------------------------------
    t1 = time.perf_counter()
    try:
        coco_evaluator.update(res)
    except Exception as e:
        print(f"[ERROR] coco_evaluator.update() failed: {e}")
        return
    evaluator_time += time.perf_counter() - t1

    if DEBUG:
        for iou_type, coco_eval_obj in coco_evaluator.coco_eval.items():
            if coco_eval_obj and coco_eval_obj.cocoDt:
                n_dt = len(coco_eval_obj.cocoDt.dataset.get("annotations", []))
                n_gt = len(coco_eval_obj.cocoGt.dataset.get("annotations", []))
                print(f"[DEBUG] {iou_type}: GT={n_gt}, DT={n_dt}")

    # ------------------------------------------------------------------
    # Summarize metrics
    # ------------------------------------------------------------------
    print("\n[INFO] Accumulating COCO metrics...")
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    
    print("[INFO] Per-class AP computation...")
    # Get COCOeval object
    coco_eval = coco_evaluator.coco_eval["bbox"]
    cat_ids = coco_eval.params.catIds
    cat_names = [coco_evaluator.coco_gt.loadCats([cid])[0]["name"] for cid in cat_ids]

    # Per-class AP (IoU=0.5:0.95)
    print("\n[INFO] Per-class AP summary (IoU=0.5:0.95):")
    per_class_ap = []
    for idx, catId in enumerate(cat_ids):
        # COCOeval doesn't expose per-class AP directly, so compute manually
        precision = coco_eval.eval["precision"][:, :, idx, 0, -1]  # IoU x recall x class x area x maxDets
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        print(f"  {idx+1:2d}. {cat_names[idx]:20s} (ID {catId:3d}): AP = {ap:.3f}")
        per_class_ap.append({"id": catId, "name": cat_names[idx], "AP": ap})

    # Convert per-class AP summary to DataFrame
    df_ap = pd.DataFrame(per_class_ap)

    # Save to CSV
    csv_path = os.path.join(vis_dir, "coco_per_class_AP.csv")
    df_ap.to_csv(csv_path, index=False)
    print(f"[INFO] Saved per-class AP summary to: {csv_path}")

    # ----------------------------------------------------------
    # Optional: Save detailed COCOeval results (global + per-class)
    # ----------------------------------------------------------
    metrics_names = [
        "AP@[.50:.95]", "AP@0.50", "AP@0.75",
        "AP@small", "AP@medium", "AP@large",
        "AR@1", "AR@10", "AR@100",
        "AR@small", "AR@medium", "AR@large"
    ]

    df_stats = pd.DataFrame({
        "metric": metrics_names,
        "value": coco_eval.stats
    })
    csv_stats = os.path.join(vis_dir, "coco_global_metrics.csv")
    df_stats.to_csv(csv_stats, index=False)
    print(f"[INFO] Saved global COCO metrics to: {csv_stats}")

    print(f"\n[INFO] Evaluation complete for {len(data_loader.dataset)} images")
    print(f"  • Avg model inference time/batch : {model_time / len(data_loader):.4f}s")
    print(f"  • Avg COCOeval update time/batch : {evaluator_time / len(data_loader):.4f}s")
    print(f"  • Visualizations saved to: {os.path.abspath(vis_dir)}")

    return coco_evaluator

# @torch.inference_mode()
# def simplemodelevaluate(
#     model,
#     data_loader,
#     device,
#     class_map=None,
#     score_thresh=0.05,
#     vis_dir="debug_vis",
#     max_vis=100,
# ):
#     """
#     Evaluate an object detection model (e.g., FasterRCNN, DETR) on a COCO-style dataset.

#     Features:
#       • Converts predictions from [x1, y1, x2, y2] → [x, y, w, h] for COCOeval
#       • Safe clamping and tensor→list conversion
#       • Per-image stats (GT count, predictions, TP/FP/FN)
#       • Two-stage visualization:
#             - BEFORE conversion (xyxy, raw detector output)
#             - AFTER conversion (xywh, COCO-style)
#       • Fully compatible with pycocotools.COCoeval
#     """

#     import os, time, torch
#     import numpy as np
#     from tqdm import tqdm

#     # --------------------------------------------------------------
#     # Initialization
#     # --------------------------------------------------------------
#     os.makedirs(vis_dir, exist_ok=True)
#     device, cpu_device = torch.device(device), torch.device("cpu")
#     model.eval().to(device)

#     print(f"\n[INFO] Starting evaluation on device: {device}")

#     # Build COCO API ground-truth object from dataset
#     coco = convert_to_coco_api_new(data_loader.dataset)
#     print("GT categories:", sorted({ann["category_id"] for ann in coco.dataset["annotations"]}))
#     coco_evaluator = CocoEvaluator(coco, ["bbox"])

#     model_time = evaluator_time = 0.0
#     progress_bar = tqdm(data_loader, desc="Evaluating", unit="batch")
#     vis_count = 0
    
#     # --------------------------------------------------------------
#     # Step 0: initialize once, outside the loader loop
#     # --------------------------------------------------------------
#     res = {}       # accumulate all predictions for all images
#     img_map = {}   # remember image tensors if you need them later

#     # --------------------------------------------------------------
#     # Step 1: main loop over DataLoader
#     # --------------------------------------------------------------
#     for images, targets in progress_bar:
#         images = [img.to(device, non_blocking=True) for img in images]

#         # ---------------- Step 1: Run inference ----------------
#         t0 = time.perf_counter()
#         outputs = model(images)
#         if device.type == "cuda":
#             torch.cuda.synchronize()
#         model_time += time.perf_counter() - t0

#         # Move model outputs back to CPU
#         outputs = [{k: v.to(cpu_device) for k, v in o.items()} for o in outputs]

#         # ---------------- Step 2: Optional label remapping ----------------
#         # (Used for COCO→Waymo or custom mappings)
#         if class_map is not None:
#             for out in outputs:
#                 labels = out["labels"]
#                 mapped = torch.zeros_like(labels)
#                 for src, dst in class_map.items():
#                     mapped[labels == src] = dst
#                 out["labels"] = mapped

#         # Storage for results and images
#         # res = {}
#         # img_map = {}

#         # --------------------------------------------------------------
#         # Process each image in this batch individually
#         # --------------------------------------------------------------
#         # --------------------------------------------------------------
#         # collect predictions for each image
#         # --------------------------------------------------------------
#         for img, target, output in zip(images, targets, outputs):
#             # Extract unique image ID
#             image_id = int(target["image_id"]) if isinstance(target, dict) else int(target[0]["image_id"])

#             # Extract ground truth boxes and predictions
#             gt_boxes = target["boxes"].cpu() #[6, 4]
#             n_gt = len(gt_boxes)
#             n_before = len(output["boxes"])

#             # Filter low-confidence detections
#             keep = output["scores"] > score_thresh if "scores" in output else torch.zeros(0, dtype=torch.bool)
#             output = {k: v[keep] for k, v in output.items()}
#             n_after = len(output["boxes"])

#             print(f"[INFO] Image {image_id}: GT={n_gt}, predicted(before)={n_before}, after filter={n_after}")

#             # Compute quick detection stats (based on IoU threshold)
#             stats = _evaluate_single_image(output["boxes"], gt_boxes)
#             print(f"        TP={stats['tp']}, FP={stats['fp']}, FN={stats['fn']}")

#             res[image_id] = output
#             img_map[image_id] = img.cpu() #img
#             target["boxes"] = gt_boxes  # ensure shape consistency

#             # ---------------- Visualization 1: BEFORE conversion ----------------
#             # Model outputs are in xyxy format (PyTorch detector default)
#             if vis_count < max_vis:
#                 vis_count += 1
#                 _visualize_prediction(
#                     img.cpu(),
#                     target,
#                     output,
#                     save_path=os.path.join(vis_dir, f"{image_id}_before_xyxy.jpg"),
#                     gt_box_type="xyxy",
#                     pred_box_type="xyxy",   # raw model output
#                 )

#         # Skip empty batches
#         if len(res) == 0:
#             print("[WARN] No predictions in this batch.")
#             continue

#         # --------------------------------------------------------------
#         # Step 3: Convert predicted boxes from xyxy → xywh (COCO format)
#         # --------------------------------------------------------------
#         for img_id, out in res.items():
#             boxes = out["boxes"]
#             if boxes.numel() == 0:
#                 continue

#             # Retrieve image size for clamping
#             img_w, img_h = img_map[img_id].shape[-1], img_map[img_id].shape[-2]

#             # Convert predictions safely
#             boxes_xywh = xyxy_to_xywh_safe(boxes, img_w, img_h)

#             # ✅ Replace with COCO-format boxes ([x, y, w, h])
#             out["boxes"] = boxes_xywh
#             res[img_id] = out

#             # ---------------- Visualization 2: AFTER conversion ----------------
#             # Here we visualize the COCO-format predictions (xywh)
#             # to confirm that conversion did not distort the boxes.
#             # Ground-truth boxes are drawn in xyxy format for consistency.
#             if vis_count < max_vis:
#                 vis_count += 1
#                 vis_path = os.path.join(vis_dir, f"{img_id}_after_xywh.jpg")
#                 _visualize_prediction(
#                     img_map[img_id].cpu(),
#                     target,
#                     out,
#                     save_path=vis_path,
#                     gt_box_type="xyxy",     # GTs remain in xyxy (dataset transform)
#                     pred_box_type="xywh",   # Predictions are now COCO-style
#                 )

#         # --------------------------------------------------------------
#         # Step 4: Convert tensors → Python lists for COCO API
#         # --------------------------------------------------------------
#         #
#         # COCOeval requires the following structure:
#         #
#         # res = {
#         #     image_id (int): {
#         #         "boxes":  [[x, y, w, h], [x, y, w, h], ...],
#         #         "scores": [float, float, ...],
#         #         "labels": [int, int, ...]
#         #     },
#         #     ...
#         # }
#         #
#         # Both GT and predictions passed into COCOeval are in xywh format.
#         # --------------------------------------------------------------
#         for img_id, out in res.items():
#             out["boxes"]  = out["boxes"].cpu().numpy().astype(np.float32)
#             out["scores"] = out["scores"].cpu().numpy().astype(np.float32)
#             out["labels"] = out["labels"].cpu().numpy().astype(np.int32)
#             res[img_id] = out

#         #new debug
#         gt_ids  = sorted({ann["category_id"] for ann in coco_evaluator.coco_gt.dataset["annotations"]})
#         dt_ids  = sorted({int(l) for r in res.values() for l in r["labels"].tolist()})
#         print("GT category ids :", gt_ids)
#         print("DT category ids :", dt_ids)
        
#         gt_img_ids = set(coco_evaluator.coco_gt.getImgIds())
#         dt_img_ids = set(res.keys())
#         print("Images missing in predictions:", gt_img_ids - dt_img_ids)

#         # --------------------------------------------------------------
#         # Step 5: Update COCO evaluator with predictions
#         # --------------------------------------------------------------
#         t1 = time.perf_counter()
#         try:
#             coco_evaluator.update(res)
#         except Exception as e:
#             print(f"[ERROR] coco_evaluator.update() failed: {e}")
#             print("Offending image IDs:", list(res.keys()))
#             continue
#         evaluator_time += time.perf_counter() - t1

#     # --------------------------------------------------------------
#     # Step 6: Summarize results (COCO metrics)
#     # --------------------------------------------------------------
#     print("\n[INFO] Accumulating COCO metrics...")
#     coco_evaluator.accumulate()
#     coco_evaluator.summarize()

#     # --------------------------------------------------------------
#     # Step 7: Timing summary
#     # --------------------------------------------------------------
#     print(f"\n[INFO] Evaluation complete for {len(data_loader.dataset)} images")
#     print(f"  • Avg model inference time/batch : {model_time / len(data_loader):.4f}s")
#     print(f"  • Avg COCOeval update time/batch : {evaluator_time / len(data_loader):.4f}s")
#     print(f"  • Visualizations saved to: {os.path.abspath(vis_dir)}")

#     return coco_evaluator


@torch.inference_mode()
def simplemodelevaluate_debug(
    model,
    data_loader,
    device,
    class_map=None,
    score_thresh=0.05,
    vis_dir="debug_vis",
    max_vis=5,
    enable_debug_iou=True,   # ← Optional deep COCOeval debug
):
    """
    Evaluate a detection model on a COCO-style dataset with automatic validation.

    Adds:
      - Bounding box format checks
      - NaN/Inf and negative size checks
      - Optional COCOeval computeIoU debug patch
    """

    import os, time, torch
    import numpy as np
    from tqdm import tqdm
    from pycocotools import cocoeval

    # ----------------------------- #
    # Optional COCOeval debug patch #
    # ----------------------------- #
    if enable_debug_iou:
        orig_computeIoU = cocoeval.COCOeval.computeIoU

        def debug_computeIoU(self, imgId, catId):
            try:
                return orig_computeIoU(self, imgId, catId)
            except Exception as e:
                print("\n[DEBUG] >>> computeIoU failed <<<")
                print(f"imgId={imgId}, catId={catId}")
                gt = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=[imgId], catIds=[catId]))
                dt = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=[imgId], catIds=[catId]))
                print(f"  GT boxes ({len(gt)}):", [g['bbox'] for g in gt][:3])
                print(f"  DT boxes ({len(dt)}):", [d['bbox'] for d in dt][:3])
                raise e

        cocoeval.COCOeval.computeIoU = debug_computeIoU
        print("[INFO] COCOeval computeIoU debug patch enabled.")

    # ----------------------------- #
    # Initialize                    #
    # ----------------------------- #
    os.makedirs(vis_dir, exist_ok=True)
    device, cpu_device = torch.device(device), torch.device("cpu")
    model.eval().to(device)

    print(f"\n[INFO] Starting evaluation on device: {device}")

    coco = convert_to_coco_api_new(data_loader.dataset)
    print("GT categories:", sorted({ann["category_id"] for ann in coco.dataset["annotations"]}))
    coco_evaluator = CocoEvaluator(coco, ["bbox"])
    #coco_evaluator.params.iouType = "bbox"  # ensure bbox IoU

    model_time = evaluator_time = 0.0
    progress_bar = tqdm(data_loader, desc="Evaluating", unit="batch")
    vis_count = 0

    # ----------------------------- #
    # Helper: validation functions  #
    # ----------------------------- #
    def check_box_array(boxes, img_id, name="boxes"):
        """
        Validate a box array before passing to COCOeval.
        Returns True if valid, otherwise prints diagnostic info and returns False.
        """
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            print(f"[ERROR] {name} for img {img_id}: wrong shape {boxes.shape}")
            return False
        if not np.isfinite(boxes).all():
            print(f"[ERROR] {name} for img {img_id}: NaN or Inf values detected.")
            return False
        if (boxes[:, 2] <= 0).any() or (boxes[:, 3] <= 0).any():
            print(f"[ERROR] {name} for img {img_id}: non-positive width/height found.")
            return False
        return True

    # ----------------------------- #
    # Main evaluation loop          #
    # ----------------------------- #
    for images, targets in progress_bar:
        images = [img.to(device, non_blocking=True) for img in images]

        # --- Model inference ---
        t0 = time.perf_counter()
        outputs = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        model_time += time.perf_counter() - t0
        outputs = [{k: v.to(cpu_device) for k, v in o.items()} for o in outputs]

        # --- Optional class remap ---
        if class_map is not None:
            for out in outputs:
                labels = out["labels"]
                mapped = torch.zeros_like(labels)
                for src, dst in class_map.items():
                    mapped[labels == src] = dst
                out["labels"] = mapped

        res = {}
        img_map = {}

        for img, target, output in zip(images, targets, outputs):
            image_id = int(target["image_id"]) if isinstance(target, dict) else int(target[0]["image_id"])
            gt_boxes = target["boxes"].cpu()
            n_gt = len(gt_boxes)
            keep = output["scores"] > score_thresh if "scores" in output else torch.zeros(0, dtype=torch.bool)
            output = {k: v[keep] for k, v in output.items()}
            n_after = len(output["boxes"])
            print(f"[INFO] Image {image_id}: GT={n_gt}, predicted(after filter)={n_after}")

            res[image_id] = output
            img_map[image_id] = img
            target["boxes"] = gt_boxes

        # --- Skip empty batch ---
        if len(res) == 0:
            continue

        # ----------------------------- #
        # Validate + convert predictions
        # ----------------------------- #
        for img_id, out in res.items():
            boxes = out["boxes"]
            if boxes.numel() == 0:
                continue

            img_w, img_h = img_map[img_id].shape[-1], img_map[img_id].shape[-2]

            # Safe conversion
            boxes_xywh = xyxy_to_xywh_safe(boxes, img_w, img_h, verbose=False)
            out["boxes"] = boxes_xywh
            res[img_id] = out

            # Validation (torch)
            arr = boxes_xywh.cpu().numpy()
            if not check_box_array(arr, img_id, "pred_boxes"):
                print(f"[ERROR] Image {img_id} failed validation, skipping.")
                res.pop(img_id)
                continue

        # ----------------------------- #
        # Convert tensors → numpy arrays
        # ----------------------------- #
        for img_id, out in res.items():
            out["boxes"]  = out["boxes"].cpu().numpy().astype(np.float32)
            out["scores"] = out["scores"].cpu().numpy().astype(np.float32)
            out["labels"] = out["labels"].cpu().numpy().astype(np.int32)
            res[img_id] = out

        # Final validation before COCOeval
        for img_id, out in res.items():
            if not check_box_array(out["boxes"], img_id, "final_pred_boxes"):
                res.pop(img_id, None)

        if len(res) == 0:
            print("[WARN] All predictions in batch invalid; skipping COCO update.")
            continue

        # ---------------------------------------------------------
        #  Validate BOTH predictions (DT) and ground-truth (GT)
        # ---------------------------------------------------------
        from pycocotools.coco import COCO

        def validate_coco_gt(coco_gt: COCO):
            bad = 0
            for ann_id, ann in coco_gt.anns.items():
                bbox = ann.get("bbox", [])
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    print(f"[BAD GT] ann_id={ann_id}: malformed bbox {bbox}")
                    bad += 1
                elif any((not np.isfinite(v)) or v < 0 for v in bbox[2:]):  # width/height check
                    print(f"[BAD GT] ann_id={ann_id}: non-finite or negative w/h {bbox}")
                    bad += 1
            if bad > 0:
                print(f"[ERROR] {bad} invalid GT annotations detected!")
            else:
                print("[INFO] All GT annotations valid.")
            return bad == 0


        def validate_res_for_update(res_dict):
            bad_imgs = []
            for img_id, out in res_dict.items():
                boxes = np.asarray(out["boxes"])
                if boxes.ndim != 2 or boxes.shape[1] != 4:
                    print(f"[ERROR] img {img_id}: wrong box shape {boxes.shape}")
                    bad_imgs.append(img_id); continue
                if not np.isfinite(boxes).all():
                    print(f"[ERROR] img {img_id}: NaN/Inf in boxes"); bad_imgs.append(img_id)
                    continue
                if (boxes[:, 2] <= 0).any() or (boxes[:, 3] <= 0).any():
                    print(f"[ERROR] img {img_id}: non-positive w/h ->",
                        boxes[(boxes[:, 2] <= 0) | (boxes[:, 3] <= 0)])
                    bad_imgs.append(img_id)
            if bad_imgs:
                print(f"[WARN] Removing {len(bad_imgs)} invalid prediction entries before update.")
                for bid in bad_imgs:
                    res_dict.pop(bid, None)
            return res_dict


        # --- validate GT once ---
        validate_coco_gt(coco_evaluator.coco_gt)

        # --- validate predictions before update ---
        res = validate_res_for_update(res)

        # ----------------------------- #
        # Update evaluator safely        #
        # ----------------------------- #
        t1 = time.perf_counter()
        try:
            coco_evaluator.update(res)
        except Exception as e:
            print(f"[ERROR] coco_evaluator.update() failed: {e}")
            print("Offending image IDs:", list(res.keys())[:10])
            # Optional: dump failing boxes for analysis
            for img_id, out in res.items():
                print(f"  img_id={img_id}, boxes sample={out['boxes'][:3]}")
            continue
        evaluator_time += time.perf_counter() - t1

    # ----------------------------- #
    # Summarize results             #
    # ----------------------------- #
    print("\n[INFO] Accumulating COCO metrics...")
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    print(f"\n[INFO] Evaluation complete for {len(data_loader.dataset)} images")
    print(f"  • Avg model inference time/batch : {model_time / len(data_loader):.4f}s")
    print(f"  • Avg COCOeval update time/batch : {evaluator_time / len(data_loader):.4f}s")
    #print(f"  • COCOeval IoU type : {coco_evaluator.params.iouType}")
    print(f"  • Visualizations saved to: {os.path.abspath(vis_dir)}")

    return coco_evaluator

def _visualize_prediction(
    img_tensor,
    target,
    output,
    save_path,
    class_names=None,
    gt_box_type="auto",    # "auto", "xyxy", or "xywh"
    pred_box_type="auto",  # "auto", "xyxy", or "xywh"
):
    """
    Visualize image with ground-truth (green) and predicted (red) boxes + class labels.

    Args:
        img_tensor (Tensor[C,H,W]): input image tensor (normalized or not)
        target (dict): ground truth, must contain "boxes", "labels"
        output (dict): model output, must contain "boxes", "labels", "scores"
        save_path (str): file path to save visualization
        class_names (dict or list, optional): id->name mapping
        gt_box_type (str): "xyxy", "xywh", or "auto" to auto-detect GT format
        pred_box_type (str): "xyxy", "xywh", or "auto" to auto-detect prediction format
    """

    # ---------- Utility: unnormalize ImageNet-normalized tensor ----------
    def unnormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        img = img.clone()
        for c, (m, s) in enumerate(zip(mean, std)):
            img[c] = img[c] * s + m
        return img.clamp(0, 1)

    # ---------- Utility: convert label id to readable text ----------
    def get_label_text(label_id, score=None):
        if torch.is_tensor(label_id):
            label_id = int(label_id.item())
        name = str(label_id)
        if class_names:
            if isinstance(class_names, dict):
                name = class_names.get(label_id, str(label_id))
            elif isinstance(class_names, (list, tuple)) and label_id < len(class_names):
                name = class_names[label_id]
        return f"{name}:{score:.2f}" if score is not None else name

    # ---------- Prepare image ----------
    img_cpu = img_tensor.detach().cpu()
    vmin, vmax = float(img_cpu.min()), float(img_cpu.max())
    if vmin < 0 or vmax > 1.2:
        img_status = "Likely normalized (ImageNet mean/std)"
        img_cpu = unnormalize(img_cpu)
    else:
        img_status = "Unnormalized or already [0,1]"
    print(f"[DEBUG] Image tensor status: {img_status}, range [{vmin:.3f}, {vmax:.3f}]")

    img = to_pil_image(img_cpu)
    img_w, img_h = img.width, img.height

    # ---------- Extract boxes ----------
    target_boxes = target.get("boxes", torch.zeros((0, 4)))
    pred_boxes = output.get("boxes", torch.zeros((0, 4)))
    pred_labels = output.get("labels", [])
    pred_scores = output.get("scores", [])

    n_gt = len(target_boxes)
    n_pred = len(pred_boxes)
    print(f"[DEBUG] Ground-truth: {n_gt}, Predictions: {n_pred}")

    # ---------- Auto-detect box format if requested ----------
    def detect_box_type(boxes):
        """Detect if boxes look like xyxy or xywh based on geometry."""
        if len(boxes) == 0:
            return "xyxy"
        b = boxes[0]
        if b[2] > b[0] and b[3] > b[1]:
            # both formats satisfy this, so check relative size
            # if w/h seem small compared to img dims -> likely xywh
            if b[0] + b[2] < img_w and b[1] + b[3] < img_h:
                return "xywh"
            return "xyxy"
        return "xyxy"

    gt_format = detect_box_type(target_boxes) if gt_box_type == "auto" else gt_box_type
    pred_format = detect_box_type(pred_boxes) if pred_box_type == "auto" else pred_box_type

    print(f"[DEBUG] GT box type={gt_format}, Pred box type={pred_format}")

    # ---------- Draw ----------
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    # --- Draw GT boxes (green) ---
    for box, label in zip(target_boxes, target.get("labels", [])):
        if len(box) != 4:
            continue
        x, y, w, h = box.tolist()
        if gt_format == "xyxy":
            x2, y2 = w, h
            w, h = x2 - x, y2 - y
        rect = patches.Rectangle((x, y), w, h,
                                 linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(x, max(y - 5, 5), get_label_text(label),
                color="white", fontsize=8, weight="bold",
                bbox=dict(facecolor="green", alpha=0.4, pad=1, edgecolor="none"))

    # --- Draw predictions (red) ---
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        if len(box) != 4:
            continue
        x, y, w, h = box.tolist()
        if pred_format == "xyxy":
            x2, y2 = w, h
            w, h = x2 - x, y2 - y
        rect = patches.Rectangle((x, y), w, h,
                                 linewidth=2, edgecolor="red", facecolor="none", alpha=0.6)
        ax.add_patch(rect)
        ax.text(x, max(y - 5, 5), get_label_text(label, score),
                color="yellow", fontsize=8, weight="bold",
                bbox=dict(facecolor="red", alpha=0.4, pad=1, edgecolor="none"))

    # ---------- Legend ----------
    legend_handles = [
        patches.Patch(color='lime', label=f'Ground Truth ({n_gt})'),
        patches.Patch(color='red',  label=f'Prediction ({n_pred})'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8,
              framealpha=0.5, facecolor='white')

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Visualization saved to {save_path}\n")

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