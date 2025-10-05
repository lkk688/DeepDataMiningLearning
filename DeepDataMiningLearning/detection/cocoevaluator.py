# ===== Option 1: Wrap CocoDetection to return a dict target per image =====
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F

# If you use the torchvision references' COCO evaluator, keep these imports:
#from coco_eval import CocoEvaluator  # pip/clone torchvision references if needed
from pycocotools.coco import COCO

import copy
import io
from contextlib import redirect_stdout

import numpy as np
import pycocotools.mask as mask_util
import torch
import utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

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

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

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
        return coco_results

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
            boxes = convert_to_xywh(boxes).tolist()
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


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

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

# -----------------------------
# Dataset wrapper (core of Option 1)
# -----------------------------
class CocoDictTarget(Dataset):
    """
    Wraps torchvision.datasets.CocoDetection so that __getitem__ returns:
        (Tensor[C,H,W], {
            "boxes": FloatTensor[N,4] (xyxy),
            "labels": LongTensor[N],
            "iscrowd": LongTensor[N],
            "area": FloatTensor[N],
            "image_id": LongTensor[1]
        })
    This works even when an image has zero annotations.
    """
    def __init__(self, root, annFile, transforms=None):
        self.ds = CocoDetection(root=root, annFile=annFile)
        self.transforms = transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, anno_list = self.ds[idx]                   # anno_list: list[dict] (may be empty)
        image_id = self.ds.ids[idx]                     # COCO image id

        # Build COCO-style dict target
        boxes, labels, iscrowd, areas = [], [], [], []
        for obj in anno_list:
            # COCO bbox is [x, y, w, h]; convert to xyxy
            x, y, w, h = obj.get("bbox", [0.0, 0.0, 0.0, 0.0])
            boxes.append([x, y, x + w, y + h])
            labels.append(int(obj.get("category_id", 1)))
            iscrowd.append(int(obj.get("iscrowd", 0)))
            areas.append(float(obj.get("area", w * h)))

        # Handle empty images gracefully
        if len(boxes) == 0:
            boxes_t   = torch.zeros((0, 4), dtype=torch.float32)
            labels_t  = torch.zeros((0,),  dtype=torch.int64)
            iscrowd_t = torch.zeros((0,),  dtype=torch.int64)
            areas_t   = torch.zeros((0,),  dtype=torch.float32)
        else:
            boxes_t   = torch.as_tensor(boxes,  dtype=torch.float32)
            labels_t  = torch.as_tensor(labels, dtype=torch.int64)
            iscrowd_t = torch.as_tensor(iscrowd, dtype=torch.int64)
            areas_t   = torch.as_tensor(areas,   dtype=torch.float32)

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "iscrowd":  iscrowd_t,
            "area":     areas_t,
            "image_id": torch.tensor(image_id, dtype=torch.int64),
        }

        # Convert PIL -> Tensor
        img = F.to_tensor(img)

        # Optional extra transforms that expect (img, dict-target)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

# -----------------------------
# COCO API bridge (fast path for CocoDetection; fallback otherwise)
# -----------------------------
def get_coco_api_from_dataset(dataset):
    """
    If the dataset (or its .ds) is a CocoDetection, return its COCO API directly.
    Otherwise, fall back to converting a dict-based dataset to COCO.
    """
    try:
        if isinstance(getattr(dataset, "ds", None), CocoDetection):
            return dataset.ds.coco
        if isinstance(dataset, CocoDetection):
            return dataset.coco
    except Exception:
        pass
    # Fallback for custom dict-target datasets
    return convert_to_coco_api(dataset)

def convert_to_coco_api(ds):
    """
    Minimal converter for dict-target datasets (not used for CocoDictTarget).
    Expects each target to have: boxes (xyxy), labels, iscrowd, area, image_id.
    """
    coco_ds = COCO()
    ann_id = 1
    coco_dict = {"images": [], "categories": [], "annotations": []}
    categories = set()

    for idx in range(len(ds)):
        img, t = ds[idx]
        image_id = int(t["image_id"].item() if torch.is_tensor(t["image_id"]) else t["image_id"])
        h, w = img.shape[-2:]
        coco_dict["images"].append({"id": image_id, "height": h, "width": w})

        # xyxy -> xywh
        boxes = t["boxes"]
        if torch.is_tensor(boxes): boxes = boxes.tolist()
        labels = t["labels"].tolist()
        areas = t["area"].tolist()
        iscrowd = t["iscrowd"].tolist()

        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b
            xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            cat_id = int(labels[i])
            ann = {
                "id": ann_id,
                "image_id": image_id,
                "bbox": xywh,
                "category_id": cat_id,
                "area": float(areas[i]),
                "iscrowd": int(iscrowd[i]),
            }
            coco_dict["annotations"].append(ann)
            categories.add(cat_id)
            ann_id += 1

    coco_dict["categories"] = [{"id": cid, "name": str(cid)} for cid in sorted(categories)]
    coco_ds.dataset = coco_dict
    coco_ds.createIndex()
    return coco_ds

# -----------------------------
# Enhanced evaluator (CUDA-safe, timed, commented)
# -----------------------------
import time
from tqdm import tqdm

@torch.inference_mode()
def simplemodelevaluate(model, data_loader, device):
    """
    Runs COCO-style evaluation on a dict-target dataset (like CocoDictTarget).
    """
    device = torch.device(device)
    cpu_device = torch.device("cpu")

    model.eval().to(device)
    print(f"\n[INFO] Evaluating on device: {device}")

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    model_time = 0.0
    eval_time = 0.0

    for images, targets in tqdm(data_loader, desc="Evaluating", unit="batch"):
        images = [img.to(device, non_blocking=True) for img in images]

        t0 = time.perf_counter()
        outputs = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        model_time += time.perf_counter() - t0

        # Move preds to CPU for pycocotools
        outputs = [{k: v.to(cpu_device) for k, v in out.items()} for out in outputs]

        # Map image_id -> prediction
        res = {}
        for tgt, out in zip(targets, outputs):
            image_id = int(tgt["image_id"].item() if torch.is_tensor(tgt["image_id"]) else tgt["image_id"])
            res[image_id] = out

        t1 = time.perf_counter()
        coco_evaluator.update(res)
        eval_time += time.perf_counter() - t1

    print("\n[INFO] Accumulating metrics...")
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    n_batches = max(1, len(data_loader))
    print(f"\n[INFO] Timing:")
    print(f"  • Avg model (GPU) time / batch : {model_time / n_batches:.4f}s")
    print(f"  • Avg eval  (CPU) time / batch : {eval_time  / n_batches:.4f}s\n")

    return coco_evaluator

# -----------------------------
# Faster R-CNN loader (pretrained COCO)
# -----------------------------
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def get_fasterrcnn_model(device="cuda"):
    # Try modern weights API; fall back to legacy 'pretrained=True'
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
    except Exception:
        model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device).eval()
    return model

# -----------------------------
# COCO val2017 DataLoader (uses Option 1 wrapper)
# -----------------------------
def get_coco_val_dataloader(coco_root, batch_size=2, workers=4):
    img_dir = f"{coco_root}/val2017"
    ann_file = f"{coco_root}/annotations/instances_val2017.json"

    dataset = CocoDictTarget(root=img_dir, annFile=ann_file, transforms=None)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=lambda batch: tuple(zip(*batch)),  # keep (images, targets) as tuples of length B
    )
    return loader

# -----------------------------
# Main (example)
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco_root = "/mnt/e/Shared/Dataset/coco2017" 

    model = get_fasterrcnn_model(device=device)
    data_loader = get_coco_val_dataloader(coco_root, batch_size=2, workers=4)

    simplemodelevaluate(model, data_loader, device)

if __name__ == "__main__":
    main()