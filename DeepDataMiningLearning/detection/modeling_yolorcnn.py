import math
from typing import List, Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")     # headless backend for PNG/PDF saving
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, nms
from PIL import Image
import torchvision.transforms as T
import matplotlib.patches as patches

# --- 1. import our wrapper builder (from the previous code) ---
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.ops import box_convert, nms
import importlib

# -----------------------------------------------------------------------------
# 🧱 Helper: Safe YOLOv8 loader (no dataset checks or downloads)
# -----------------------------------------------------------------------------
def _safe_open_yolo(weights="yolov8n.pt"):
    """
    Load Ultralytics YOLOv8 weights safely without triggering dataset downloads.
    Handles both old and new Ultralytics versions (args: dict or Namespace).
    """
    from ultralytics import YOLO
    y = YOLO(weights)

    # Disable dataset validation / training hooks
    args = getattr(y.model, "args", None)
    if args is not None:
        # Handle both dict and namespace forms
        if isinstance(args, dict):
            args["data"] = None
        else:
            setattr(args, "data", None)

    # Disable overrides dataset reference if present
    if hasattr(y, "overrides") and isinstance(y.overrides, dict):
        y.overrides["data"] = None

    # Remove trainer references if they exist
    if hasattr(y, "trainer"):
        y.trainer = None

    return y


# -----------------------------------------------------------------------------
# 🧩 YoloRCNN (Unified YOLOv8 + Torchvision-Compatible Wrapper)
# -----------------------------------------------------------------------------
class YoloRCNN(nn.Module):
    def __init__(self,
                 weights: str = "yolov8n.pt",
                 use_ultralytics_loss: bool = True,
                 conf_thresh: float = 0.25,
                 iou_thresh: float = 0.6):
        super().__init__()
        yolo = _safe_open_yolo(weights)
        self.ultra = yolo
        self.core = yolo.model

        # --- split backbone / neck / head using YAML lengths ---
        cfg = getattr(self.core, "yaml", None)
        if not (isinstance(cfg, dict) and "backbone" in cfg and "head" in cfg):
            raise RuntimeError("Could not read YOLO YAML to split backbone/neck/head.")

        n_backbone = len(cfg["backbone"])
        n_head = len(cfg["head"])

        children = list(self.core.model.children())
        assert len(children) == (n_backbone + n_head), \
            f"Mismatch: modules={len(children)} vs backbone+head={n_backbone+n_head}"

        self.backbone = nn.Sequential(*children[:n_backbone])   # original backbone (Conv, C2f, SPPF, ...)
        self.neck     = nn.Sequential(*children[n_backbone:-1]) # FPN/PAN etc. from the head block
        self.head     = children[-1]                             # final Detect layer

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # optional: Ultralytics DetectionLoss (version-agnostic import)
        self.criterion = None
        if use_ultralytics_loss:
            loss_candidates = [
                "ultralytics.utils.loss.v8DetectionLoss",          # new (>= 8.3)
                "ultralytics.models.yolo.detect.loss.DetectionLoss",  # mid (8.2)
                "ultralytics.utils.loss.DetectionLoss",               # legacy (<= 8.1)
            ]
            for path in loss_candidates:
                try:
                    module_name, cls_name = path.rsplit(".", 1)
                    mod = importlib.import_module(module_name)
                    DetLoss = getattr(mod, cls_name)
                    self.criterion = DetLoss(self.core)
                    print(f"[YoloRCNN] Using loss: {path}")
                    break
                except Exception:
                    continue
            if self.criterion is None:
                print("[Warn] Ultralytics DetectionLoss/v8DetectionLoss not found — continuing without loss.")

    # ------------------------ helpers: conversions & postproc -------------------
    def _targets_xyxy_to_yolo(self, targets, img_h, img_w):
        y = []
        for i, t in enumerate(targets):
            if len(t["boxes"]) == 0:
                continue
            boxes = box_convert(t["boxes"], in_fmt="xyxy", out_fmt="cxcywh").clone()
            boxes[:, [0, 2]] /= img_w
            boxes[:, [1, 3]] /= img_h
            labels = t["labels"].float().unsqueeze(1)
            img_idx = torch.full((len(labels), 1), i, device=boxes.device)
            y.append(torch.cat([img_idx, labels, boxes], dim=1))
        return torch.cat(y, 0) if y else torch.zeros((0, 6), device=targets[0]["boxes"].device)

    def _ultra_preds_to_torchvision(self, preds):
        out = []
        for p in preds:
            if p is None or p.numel() == 0:
                out.append({"boxes": torch.zeros((0,4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int64)})
                continue
            boxes, scores, labels = p[:, :4], p[:, 4], p[:, 5].to(torch.int64)
            keep = scores >= self.conf_thresh
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if boxes.numel():
                keep_idx = nms(boxes, scores, self.iou_thresh)
                boxes, scores, labels = boxes[keep_idx], scores[keep_idx], labels[keep_idx]
            out.append({"boxes": boxes, "scores": scores, "labels": labels})
        return out

    # ------------------------------- forward -----------------------------------
    def forward(self, images, targets=None):
        x = torch.stack(images) if isinstance(images, list) else images
        B, _, H, W = x.shape

        if self.training:
            if targets is None:
                raise ValueError("Targets required during training.")
            y = self._targets_xyxy_to_yolo(targets, H, W)
            feats = self.backbone(x)    # list of feature maps
            neck_out = self.neck(feats) # list of FPN/PAN maps
            preds = self.head(neck_out) # raw detect outputs or decoded per-version

            if self.criterion is not None and y.numel():
                total_loss, li = self.criterion(preds, y)
                lbox = li[0] if len(li) > 0 else torch.tensor(0., device=x.device)
                lcls = li[1] if len(li) > 1 else torch.tensor(0., device=x.device)
                ldfl = li[2] if len(li) > 2 else torch.tensor(0., device=x.device)
                return {
                    "loss_classifier": lcls,
                    "loss_box_reg": lbox,
                    "loss_objectness": torch.tensor(0.0, device=x.device),
                    "loss_rpn_box_reg": ldfl,
                    "loss_total": total_loss
                }
            else:
                zero = torch.tensor(0.0, device=x.device, requires_grad=True)
                return {k: zero for k in ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg", "loss_total"]}

        # eval: Ultralytics Detect head returns per-image detections in recent versions
        with torch.no_grad():
            results = self.core(x)  # keep Ultralytics eval path for decoding/NMS
            return self._ultra_preds_to_torchvision(results)

    # handy if you want features for a second-stage RCNN
    def forward_features(self, x):
        return self.neck(self.backbone(x))  # list of FPN maps

if __name__ == "__main__":
    
    # --- load pretrained YOLOv8 and wrap it ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloRCNN("yolov8n.pt").to(device)
    print("backbone layers:", len(list(model.backbone.children())))
    print("neck layers:", len(list(model.neck.children())))
    print("head:", model.head.__class__.__name__)  # Detect

    print(model.backbone)
    print(model.neck)
    print(model.head)

    # --- load and preprocess one image ---
    # You can replace this with any image path
    img_path = "sampledata/bus.jpg"

    # open image and transform
    image = Image.open(img_path).convert("RGB")

    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),  # converts to [0,1], shape [3,H,W]
    ])
    img_tensor = transform(image).to(device)

    # Torchvision-compatible input format: list of tensors
    images = [img_tensor]

    # --- 4. run inference ---
    # Forward in eval mode
    model.eval()
    with torch.no_grad():
        outputs = model(images)  # returns list[dict]: {"boxes", "labels", "scores"}

    # --- 5. visualize results ---
    output = outputs[0]
    print(outputs[0].keys())
    boxes = output["boxes"].cpu()
    scores = output["scores"].cpu()
    labels = output["labels"].cpu()

    # Optionally: confidence threshold
    conf_thresh = 0.1
    keep = scores > conf_thresh
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    # --- 6. plot image + boxes ---
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    # get class names from YOLO model
    class_names = getattr(model.core.head, "names", None) or getattr(model.core.head, "nc", None)
    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in range(len(class_names))]
    elif isinstance(class_names, int):
        # fallback to numeric labels
        class_names = [f"class_{i}" for i in range(class_names)]

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)
        label_str = class_names[label] if class_names and label < len(class_names) else str(label.item())
        ax.text(x1, y1 - 5, f"{label_str} {score:.2f}", color="yellow", fontsize=10, backgroundcolor="black")

    ax.axis("off")
    plt.title("YOLOv8 (Torchvision-compatible wrapper) Detection")
    #plt.show()
    fig.savefig("output/yolov8_detection.jpg", dpi=150)
    
    #Extract YOLO features for an RCNN head
    x = torch.randn(1, 3, 640, 640, device=device)
    features = model.forward_features(x)
    print([f.shape for f in features])
    