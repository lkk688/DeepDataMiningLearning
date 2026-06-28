"""
ngdet.train
===========

A single, **modular, teaching-oriented** training entry point that shows the TWO
ways students should know how to train a detector:

  1. A **raw PyTorch training loop** (`train_loop`) — you see every step: forward →
     loss → backward → optimizer step → LR schedule → checkpoint. One loop drives
     two very different model families via a small `TrainBackend` abstraction:
       * `torchvision` — Faster R-CNN / RetinaNet / FCOS  (the "FasterRCNN style":
         `loss_dict = model(images, targets)`), cf. detection/mytrainv2.py
       * `yolo`        — Ultralytics YOLO  (the "YOLO style":
         `loss, items = model.loss(batch)`), cf. detection/mytrain_yolo.py
  2. The **HuggingFace `Trainer`** (`train_hf`) — the modern, batteries-included
     path for HF models (DETR / RT-DETR), with `TrainingArguments` + a data collator.

Data comes from the same `EvalDataset` the evaluation harness uses, so any dataset
(coco/kitti/waymo/nuscenes/nuimages) and the configurable unified taxonomy work here
too. Each backend has its own collate that converts our canonical
`(PIL image, boxes_xyxy_abs, unified_labels)` sample into that model's batch format.

This is intentionally compact and readable (teaching first), not a production trainer.
For large-scale/production YOLO training prefer Ultralytics' own `model.train()`.

Run `python -m DeepDataMiningLearning.ngdet.train --help`.
"""

from __future__ import annotations
import argparse
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .taxonomy import Taxonomy
from .datasets import EvalDataset


# ===========================================================================
# Config
# ===========================================================================
@dataclass
class TrainConfig:
    # what / where
    trainer: str = "pytorch"            # "pytorch" (raw loop) | "hf" (HF Trainer)
    backend: str = "torchvision"        # raw-loop backend: "torchvision" | "yolo"
    yolo_trainer: str = "raw"           # for backend=yolo: "raw" (our loop) | "native" (ultralytics)
    model_name: str = "fasterrcnn_resnet50_fpn_v2"
    dataset: str = "nuimages"
    root: str = "/mnt/e/Shared/Dataset/NuScenes/nuimages"
    taxonomy: str = "driving3"
    nuimages_version: str = "v1.0-train"
    # validation (HF Trainer reports COCO mAP on this during training)
    eval_version: str = "v1.0-val"      # nuimages val split for the mAP callback
    eval_max_images: int = 100
    # data
    max_images: Optional[int] = 400     # cap for quick teaching runs (None = all)
    image_size: int = 640               # square train size for the YOLO backend
    # optimization
    epochs: int = 5
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    amp: bool = True
    num_workers: int = 4
    device: str = "cuda"
    # anti-forgetting knobs (§18): preserve pretrained features when fine-tuning on
    # small target data, so the easy domain doesn't overwrite the hard ones.
    trainable_backbone_layers: Optional[int] = None  # torchvision: 0=freeze backbone..5=all
    lr_backbone_mult: float = 1.0       # torchvision/HF: backbone LR = lr * this (e.g. 0.1)
    freeze_backbone: bool = False       # HF (RT-DETR): freeze the backbone entirely
    freeze: int = 0                     # YOLO native: freeze the first N layers
    # io
    output_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "train"))
    log_every: int = 20


# ===========================================================================
# Data — wrap EvalDataset into a canonical (PIL, boxes_xyxy, labels) Dataset
# ===========================================================================
class CanonicalDetData(Dataset):
    """Yields `(PIL.Image RGB, boxes [N,4] xyxy abs px, labels [N] unified ids)`.

    All backends collate from this one canonical sample, so adding a dataset (via
    EvalDataset) or a backend stays decoupled.
    """

    def __init__(self, cfg: TrainConfig, taxonomy: Taxonomy, split: Optional[str] = None):
        kw = {}
        if cfg.dataset == "nuimages":
            kw["version"] = cfg.nuimages_version
        if cfg.dataset == "mixed":
            kw["split"] = split or "train"     # train split for training
        self.ds = EvalDataset(cfg.dataset, cfg.root, taxonomy,
                              max_images=cfg.max_images, **kw)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        s = self.ds[i]
        return s.image, s.gt_boxes.astype(np.float32), s.gt_labels.astype(np.int64)


# ===========================================================================
# Backend abstraction for the RAW PyTorch loop
# ===========================================================================
class TrainBackend:
    """A trainable model + its collate + its loss step. The raw loop is agnostic to it."""
    def collate(self, samples):                      # list[(pil, boxes, labels)] -> batch
        raise NotImplementedError
    def to(self, device): self.model.to(device); return self
    def train_mode(self): self.model.train()
    def parameters(self): return self.model.parameters()
    def training_step(self, batch, device) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return (scalar loss tensor, {name: value} for logging)."""
        raise NotImplementedError
    def build_optimizer(self, cfg):
        """Default: AdamW over all trainable params. Backends override for
        discriminative (per-group) learning rates."""
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad],
                                 lr=cfg.lr, weight_decay=cfg.weight_decay)
    def save(self, path): torch.save(self.model.state_dict(), path)


class TorchvisionBackend(TrainBackend):
    """Faster R-CNN / RetinaNet / FCOS — the classic 'FasterRCNN style':
    `model(images, targets)` returns a dict of losses we sum. (cf. mytrainv2.py)"""

    def __init__(self, cfg: TrainConfig, num_classes: int):
        from torchvision.models import get_model
        from torchvision.transforms.functional import to_tensor
        self._to_tensor = to_tensor
        self.cfg = cfg
        # +1 for the background class (torchvision detection convention). We build
        # with pretrained BACKBONE but a fresh detection head sized to our classes.
        # trainable_backbone_layers (0..5) controls how much of the ResNet backbone is
        # fine-tuned: 0 freezes it entirely (anti-forgetting), 5 trains all. None=default(3).
        kw = {}
        if cfg.trainable_backbone_layers is not None:
            kw["trainable_backbone_layers"] = cfg.trainable_backbone_layers
        self.model = get_model(cfg.model_name, weights=None,
                               weights_backbone="DEFAULT", num_classes=num_classes + 1, **kw)

    def build_optimizer(self, cfg):
        # Discriminative LR: a smaller LR on the pretrained backbone preserves its
        # features (less forgetting) while the head adapts at the full LR.
        if cfg.lr_backbone_mult != 1.0:
            bb, head = [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                (bb if "backbone" in n else head).append(p)
            return torch.optim.AdamW(
                [{"params": bb, "lr": cfg.lr * cfg.lr_backbone_mult},
                 {"params": head, "lr": cfg.lr}], lr=cfg.lr, weight_decay=cfg.weight_decay)
        return super().build_optimizer(cfg)

    def collate(self, samples):
        # images: list of CHW float[0,1] tensors; targets: list of {boxes, labels}.
        images = [self._to_tensor(pil) for pil, _, _ in samples]
        targets = []
        for _, boxes, labels in samples:
            # drop degenerate boxes; shift labels by +1 (0 = background)
            b = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            l = torch.as_tensor(labels, dtype=torch.int64).reshape(-1) + 1
            keep = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
            targets.append({"boxes": b[keep], "labels": l[keep]})
        return images, targets

    def training_step(self, batch, device):
        images, targets = batch
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)          # train mode -> dict of losses
        loss = sum(loss_dict.values())
        return loss, {k: float(v) for k, v in loss_dict.items()}


# YOLO keeps its pretrained 80-class COCO head; we map unified ids -> a representative
# COCO id so the loss works WITHOUT head surgery (teaching the loop mechanics). For
# real fine-tuning to K classes, use Ultralytics `model.train()` which resets the head.
_UNIFIED_TO_COCO = {0: 2, 1: 0, 2: 1}     # vehicle->car, person->person, cyclist->bicycle


class YoloBackend(TrainBackend):
    """Ultralytics YOLO — the 'YOLO style': `model.loss(batch)` returns (loss, items)
    over box(CIoU)/cls(BCE)/dfl. (cf. mytrain_yolo.py)"""

    def __init__(self, cfg: TrainConfig, num_classes: int):
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg
        from ultralytics.utils import DEFAULT_CFG
        self.imgsz = cfg.image_size
        yolo = YOLO(cfg.model_name)
        self.model = yolo.model                          # DetectionModel (nn.Module)
        # the loss criterion reads gains from model.args
        self.model.args = get_cfg(DEFAULT_CFG, overrides={"box": 7.5, "cls": 0.5, "dfl": 1.5})
        # Ultralytics loads checkpoints for INFERENCE (params frozen); re-enable
        # gradients and float32 so the raw training loop can backprop.
        self.model.float()
        for p in self.model.parameters():
            p.requires_grad_(True)

    def collate(self, samples):
        import torch.nn.functional as F
        from torchvision.transforms.functional import to_tensor, resize
        imgs, cls, bboxes, batch_idx = [], [], [], []
        for bi, (pil, boxes, labels) in enumerate(samples):
            w0, h0 = pil.size
            img = resize(to_tensor(pil), [self.imgsz, self.imgsz])   # square resize [0,1]
            imgs.append(img)
            for (x1, y1, x2, y2), lab in zip(boxes, labels):
                # normalized cx,cy,w,h in [0,1] (resize is a bijection in normalized space)
                cx = ((x1 + x2) / 2) / w0; cy = ((y1 + y2) / 2) / h0
                bw = (x2 - x1) / w0; bh = (y2 - y1) / h0
                if bw <= 0 or bh <= 0:
                    continue
                bboxes.append([cx, cy, bw, bh])
                cls.append([_UNIFIED_TO_COCO.get(int(lab), 0)])
                batch_idx.append(bi)
        return {
            "img": torch.stack(imgs),
            "cls": torch.tensor(cls, dtype=torch.float32).reshape(-1, 1),
            "bboxes": torch.tensor(bboxes, dtype=torch.float32).reshape(-1, 4),
            "batch_idx": torch.tensor(batch_idx, dtype=torch.float32),
        }

    def training_step(self, batch, device):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, items = self.model.loss(batch)             # (sum tensor, [box, cls, dfl])
        names = ["box", "cls", "dfl"]
        return loss.sum(), {n: float(v) for n, v in zip(names, items)}


def build_backend(cfg: TrainConfig, num_classes: int) -> TrainBackend:
    return {"torchvision": TorchvisionBackend,
            "yolo": YoloBackend}[cfg.backend](cfg, num_classes)


# ===========================================================================
# The RAW PyTorch training loop (backend-agnostic)
# ===========================================================================
def train_loop(cfg: TrainConfig, backend: TrainBackend, loader: DataLoader):
    device = cfg.device
    backend.to(device)
    optimizer = backend.build_optimizer(cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.startswith("cuda"))
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"[train_loop] backend={cfg.backend} model={cfg.model_name} "
          f"epochs={cfg.epochs} steps/epoch={len(loader)} device={device}")
    for epoch in range(cfg.epochs):
        backend.train_mode()
        t0, running = time.time(), 0.0
        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss, parts = backend.training_step(batch, device)
            if not math.isfinite(float(loss)):
                print(f"  non-finite loss at step {step}: {parts}; skipping"); continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(backend.parameters(), 10.0)
            scaler.step(optimizer); scaler.update()
            running += float(loss)
            if step % cfg.log_every == 0:
                pstr = " ".join(f"{k}={v:.3f}" for k, v in parts.items())
                print(f"  e{epoch} s{step}/{len(loader)} loss={float(loss):.3f} {pstr} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}")
        sched.step()
        ckpt = os.path.join(cfg.output_dir, f"{cfg.backend}_{cfg.model_name}_e{epoch}.pt")
        backend.save(ckpt)
        print(f"[epoch {epoch}] avg_loss={running/max(1,len(loader)):.3f} "
              f"time={time.time()-t0:.0f}s -> {ckpt}")
    print("[train_loop] done.")


# ===========================================================================
# HuggingFace Trainer path (DETR / RT-DETR) — the modern batteries-included way
# ===========================================================================
class COCOmAPCallback:
    """A TrainerCallback that runs inference on a held-out split after each epoch and
    logs **COCO mAP** (via our pycocotools `COCOUnifiedEvaluator`). The model being
    trained already has `num_labels = taxonomy.num_classes`, so its predicted ids ARE
    unified ids — we can evaluate directly. Defined via a factory so `transformers` is
    only imported when the HF path actually runs."""

    @staticmethod
    def make(processor, eval_data, taxonomy, device, max_eval):
        from transformers import TrainerCallback
        from .evaluator import COCOUnifiedEvaluator
        from .detectors.base import Detection

        class _Cb(TrainerCallback):
            def on_epoch_end(self, args, state, control, model=None, **kw):
                if model is None:
                    return
                was_training = model.training
                model.eval()
                ev = COCOUnifiedEvaluator(taxonomy)
                n = min(len(eval_data), max_eval)
                with torch.no_grad():
                    for i in range(n):
                        s = eval_data[i]
                        inp = processor(images=s.image, return_tensors="pt").to(device)
                        out = model(**inp)
                        w, h = s.image.size
                        r = processor.post_process_object_detection(
                            out, threshold=0.0,
                            target_sizes=torch.tensor([[h, w]], device=device))[0]
                        det = Detection(
                            boxes=r["boxes"].detach().cpu().numpy().reshape(-1, 4),
                            scores=r["scores"].detach().cpu().numpy().reshape(-1),
                            labels=r["labels"].detach().cpu().numpy().reshape(-1),
                            names=[taxonomy.classes[int(l)] for l in r["labels"]])
                        ev.add(s, det)
                m = ev.summarize(verbose=False)
                print(f"[val @ epoch {state.epoch:.0f}] mAP={m['mAP']:.3f} "
                      f"AP50={m['AP50']:.3f} AP75={m['AP75']:.3f} (n={n} imgs)")
                # surface in the Trainer log history too
                state.log_history.append({"epoch": state.epoch, "eval_mAP": m["mAP"],
                                          "eval_AP50": m["AP50"], "eval_AP75": m["AP75"]})
                if was_training:
                    model.train()
        return _Cb()


def train_hf(cfg: TrainConfig, taxonomy: Taxonomy):
    from transformers import (AutoImageProcessor, AutoModelForObjectDetection,
                              Trainer, TrainingArguments)

    id2label = {i: n for i, n in enumerate(taxonomy.classes)}
    label2id = {n: i for i, n in id2label.items()}
    processor = AutoImageProcessor.from_pretrained(cfg.model_name)
    model = AutoModelForObjectDetection.from_pretrained(
        cfg.model_name, num_labels=taxonomy.num_classes,
        id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)

    if cfg.freeze_backbone:               # preserve pretrained backbone features
        n_frozen = 0
        for name, p in model.named_parameters():
            if "backbone" in name:
                p.requires_grad_(False)
                n_frozen += 1
        print(f"[train_hf] froze {n_frozen} backbone params (anti-forgetting)")

    data = CanonicalDetData(cfg, taxonomy)

    # Held-out validation set for the COCO-mAP callback: the mixed dataset's "val"
    # split, or the nuImages val split.
    val_cfg = TrainConfig(**{**cfg.__dict__, "nuimages_version": cfg.eval_version,
                             "max_images": cfg.eval_max_images})
    try:
        val_split = "val" if cfg.dataset == "mixed" else None
        val_data = CanonicalDetData(val_cfg, taxonomy, split=val_split).ds
    except Exception as e:  # noqa: BLE001
        print(f"[train_hf] no val split ({e}); skipping mAP callback")
        val_data = None

    def collate(samples):
        # Build COCO-style annotations and let the image processor produce the
        # DETR targets (boxes get normalized cxcywh + class labels internally).
        images, coco = [], []
        for i, (pil, boxes, labels) in enumerate(samples):
            images.append(pil)
            anns = []
            for (x1, y1, x2, y2), lab in zip(boxes, labels):
                w, h = float(x2 - x1), float(y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                anns.append({"image_id": i, "category_id": int(lab),
                             "bbox": [float(x1), float(y1), w, h],
                             "area": w * h, "iscrowd": 0})
            coco.append({"image_id": i, "annotations": anns})
        enc = processor(images=images, annotations=coco, return_tensors="pt")
        return {"pixel_values": enc["pixel_values"],
                "pixel_mask": enc.get("pixel_mask"),
                "labels": enc["labels"]}

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        # DETR-family is numerically sensitive: fp16 makes the GIoU loss go NaN, so
        # use bf16 (stable dynamic range on Ampere+); fall back to fp32. RT-DETR
        # trains stably here; plain DETR needs a much longer schedule + per-group LRs.
        bf16=cfg.amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        dataloader_num_workers=cfg.num_workers,
        logging_steps=cfg.log_every,
        save_strategy="epoch",
        remove_unused_columns=False,        # we feed pixel_values/labels directly
        report_to="none",
    )
    callbacks = []
    if val_data is not None:
        callbacks.append(COCOmAPCallback.make(
            processor, val_data, taxonomy, cfg.device, cfg.eval_max_images))

    trainer = Trainer(model=model, args=args, train_dataset=data,
                      data_collator=collate, processing_class=processor,
                      callbacks=callbacks)
    print(f"[train_hf] HF Trainer: model={cfg.model_name} "
          f"classes={taxonomy.classes} steps/epoch≈{len(data)//cfg.batch_size} "
          f"val={'%d imgs' % len(val_data) if val_data is not None else 'none'}")
    trainer.train()
    final_dir = os.path.join(cfg.output_dir, "hf_final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)    # so the eval harness can reload it
    print(f"[train_hf] done -> {final_dir}")


# ===========================================================================
# Grounding DINO fine-tuning (open-vocabulary, text-prompted)
# ===========================================================================
def _gdino_postprocess(processor, outputs, prompt, w, h, taxonomy, device):
    """Robustly post-process Grounding DINO outputs -> unified Detection (handles
    transformers signature/key variants)."""
    import numpy as np
    from .detectors.base import Detection
    ts = [(h, w)]
    # box threshold low (mAP ranks all boxes); text_threshold ~0.25 so each box gets a
    # CLEAN class phrase -- at text_threshold=0 the label becomes the whole prompt.
    try:
        res = processor.post_process_grounded_object_detection(
            outputs, threshold=0.05, text_threshold=0.25, target_sizes=ts)[0]
    except TypeError:
        res = processor.post_process_grounded_object_detection(
            outputs, threshold=0.05, target_sizes=ts)[0]
    boxes = res["boxes"].detach().cpu().numpy().reshape(-1, 4)
    scores = res["scores"].detach().cpu().numpy().reshape(-1)
    labs = res.get("text_labels", res.get("labels"))
    labs = labs.tolist() if hasattr(labs, "tolist") else labs

    def _to_uid(lab):
        if isinstance(lab, (int, np.integer)):
            return None
        uid = taxonomy.name_to_id(lab)
        if uid is not None:
            return uid
        for u, name in enumerate(taxonomy.classes):    # substring fallback
            if name.lower() in str(lab).lower():
                return u
        return None

    kb, ks, kl, kn = [], [], [], []
    for b, s, lab in zip(boxes, scores, labs):
        uid = _to_uid(lab)
        if uid is None:
            continue
        kb.append(b); ks.append(float(s)); kl.append(uid); kn.append(taxonomy.classes[uid])
    return Detection(np.asarray(kb, np.float32).reshape(-1, 4),
                     np.asarray(ks, np.float32), np.asarray(kl, np.int64), kn)


def train_gdino(cfg: TrainConfig, taxonomy: Taxonomy):
    """Fine-tune Grounding DINO (open-vocab) on our data. Unlike DETR, the prompt
    text is part of the input and the loss aligns queries to the prompt's class
    phrases (`build_label_maps` segments by '.'), so `class_labels` = unified id
    when the prompt is the class names in order."""
    from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                              Trainer, TrainingArguments, TrainerCallback)
    from .evaluator import COCOUnifiedEvaluator

    processor = AutoProcessor.from_pretrained(cfg.model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.model_name)
    if cfg.freeze_backbone:
        nf = sum(p.requires_grad_(False).numel() for n, p in model.named_parameters()
                 if "backbone" in n)
        print(f"[train_gdino] froze backbone params ({nf/1e6:.1f}M)")

    # prompt phrases = unified class names, in order -> class index aligns to phrase
    prompt = ". ".join(c.lower() for c in taxonomy.classes) + "."
    data = CanonicalDetData(cfg, taxonomy,
                            split="train" if cfg.dataset == "mixed" else None)

    def collate(samples):
        images = [s[0] for s in samples]
        enc = dict(processor(images=images, text=[prompt] * len(images),
                             return_tensors="pt", padding=True))
        labels = []
        for pil, boxes, labs in samples:
            w, h = pil.size
            bb, cl = [], []
            for (x1, y1, x2, y2), l in zip(boxes, labs):
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                if bw > 0 and bh > 0:
                    bb.append([((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h, bw, bh])
                    cl.append(int(l))
            labels.append({"class_labels": torch.tensor(cl, dtype=torch.long),
                           "boxes": torch.tensor(bb, dtype=torch.float32).reshape(-1, 4)})
        enc["labels"] = labels
        return enc

    val_split = "val" if cfg.dataset == "mixed" else None
    try:
        vcfg = TrainConfig(**{**cfg.__dict__, "nuimages_version": cfg.eval_version,
                              "max_images": cfg.eval_max_images})
        val_data = CanonicalDetData(vcfg, taxonomy, split=val_split).ds
    except Exception as e:  # noqa: BLE001
        print(f"[train_gdino] no val ({e})"); val_data = None

    class Cb(TrainerCallback):
        def on_epoch_end(self, args, state, control, model=None, **kw):
            if model is None or val_data is None:
                return
            was = model.training; model.eval()
            ev = COCOUnifiedEvaluator(taxonomy)
            n = min(len(val_data), cfg.eval_max_images)
            with torch.no_grad():
                for i in range(n):
                    s = val_data[i]; w, h = s.image.size
                    inp = processor(images=s.image, text=prompt,
                                    return_tensors="pt").to(cfg.device)
                    ev.add(s, _gdino_postprocess(processor, model(**inp), prompt,
                                                 w, h, taxonomy, cfg.device))
            m = ev.summarize(verbose=False)
            print(f"[val @ epoch {state.epoch:.0f}] mAP={m['mAP']:.3f} "
                  f"AP50={m['AP50']:.3f} (n={n})")
            if was:
                model.train()

    args = TrainingArguments(
        output_dir=cfg.output_dir, per_device_train_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs, learning_rate=cfg.lr, weight_decay=cfg.weight_decay,
        bf16=cfg.amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        dataloader_num_workers=cfg.num_workers, logging_steps=cfg.log_every,
        save_strategy="epoch", remove_unused_columns=False, report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=data, data_collator=collate,
                      processing_class=processor,
                      callbacks=[Cb()] if val_data is not None else [])
    print(f"[train_gdino] {cfg.model_name} prompt='{prompt}' "
          f"steps/epoch≈{len(data)//cfg.batch_size}")
    trainer.train()
    fd = os.path.join(cfg.output_dir, "hf_final")
    trainer.save_model(fd); processor.save_pretrained(fd)
    print(f"[train_gdino] done -> {fd}")


# ===========================================================================
# Qwen2.5-VL LoRA fine-tuning (a STANDARD grounding VLM -> clean PEFT SFT)
# ===========================================================================
def train_qwen_lora(cfg: TrainConfig, taxonomy: Taxonomy):
    """LoRA fine-tune Qwen2.5-VL to detect our classes by generating JSON boxes.
    Qwen2.5-VL (unlike LocateAnything) is a standard HF model whose `forward` computes
    a normal LM loss, so PEFT + a masked-target collate just works. The target is the
    GT boxes as JSON in the model's smart-resized coordinate space."""
    import json as _json
    from transformers import AutoProcessor, Trainer, TrainingArguments
    from qwen_vl_utils import process_vision_info, smart_resize
    from peft import LoraConfig, get_peft_model

    if "2.5" in cfg.model_name or "2_5" in cfg.model_name:
        from transformers import Qwen2_5_VLForConditionalGeneration as VL
    else:
        from transformers import Qwen2VLForConditionalGeneration as VL
    # cap vision tokens (a 1920x1280 Waymo frame is otherwise huge/slow/OOM)
    proc = AutoProcessor.from_pretrained(cfg.model_name, max_pixels=1024 * 1024)
    model = VL.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16)
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    # gradient checkpointing keeps a 3B VLM + image activations inside 24GB
    model.enable_input_require_grads()
    model.config.use_cache = False

    ip = proc.image_processor
    factor = ip.patch_size * ip.merge_size               # 28 for Qwen2.5-VL
    terms_by_class = taxonomy.open_vocab_terms_by_class()
    rep = {uid: terms[0] for uid, terms in terms_by_class.items()}   # car/person/bicycle
    prompt = ("Detect all " + ", ".join(t for t, _ in taxonomy.open_vocab_terms()) +
              ' in the image. Output ONLY a JSON list, each item '
              '{"bbox_2d":[x1,y1,x2,y2],"label":"<class>"}.')

    data = CanonicalDetData(cfg, taxonomy,
                            split="train" if cfg.dataset == "mixed" else None)

    def _target_json(boxes, labs, w, h, rw, rh):
        items = [{"bbox_2d": [round(x1 * rw / w), round(y1 * rh / h),
                              round(x2 * rw / w), round(y2 * rh / h)],
                  "label": rep[int(l)]}
                 for (x1, y1, x2, y2), l in zip(boxes, labs)]
        return _json.dumps(items)

    def collate(samples):
        input_ids, labels_list, pix, grids = [], [], [], []
        for pil, boxes, labs in samples:
            w, h = pil.size
            rh, rw = smart_resize(h, w, factor=factor,
                                  min_pixels=ip.min_pixels, max_pixels=ip.max_pixels)
            tgt = _target_json(boxes, labs, w, h, rw, rh)
            user = [{"role": "user", "content": [
                {"type": "image", "image": pil}, {"type": "text", "text": prompt}]}]
            full_msgs = user + [{"role": "assistant",
                                 "content": [{"type": "text", "text": tgt}]}]
            ptext = proc.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
            ftext = proc.apply_chat_template(full_msgs, tokenize=False)
            imgs, vids = process_vision_info(full_msgs)
            f = proc(text=[ftext], images=imgs, videos=vids, return_tensors="pt")
            p = proc(text=[ptext], images=imgs, videos=vids, return_tensors="pt")
            plen = p["input_ids"].shape[1]
            ids = f["input_ids"][0]
            lab = ids.clone(); lab[:plen] = -100        # supervise only the JSON answer
            input_ids.append(ids); labels_list.append(lab)
            pix.append(f["pixel_values"]); grids.append(f["image_grid_thw"])
        # left/right pad to the longest sequence
        maxlen = max(x.shape[0] for x in input_ids)
        pad_id = proc.tokenizer.pad_token_id or 0
        ii = torch.full((len(input_ids), maxlen), pad_id, dtype=torch.long)
        ll = torch.full((len(input_ids), maxlen), -100, dtype=torch.long)
        am = torch.zeros((len(input_ids), maxlen), dtype=torch.long)
        for i, (x, y) in enumerate(zip(input_ids, labels_list)):
            ii[i, :x.shape[0]] = x; ll[i, :y.shape[0]] = y; am[i, :x.shape[0]] = 1
        return {"input_ids": ii, "attention_mask": am, "labels": ll,
                "pixel_values": torch.cat(pix), "image_grid_thw": torch.cat(grids)}

    args = TrainingArguments(
        output_dir=cfg.output_dir, per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=max(1, 8 // cfg.batch_size),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr, weight_decay=cfg.weight_decay,
        bf16=cfg.amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        dataloader_num_workers=cfg.num_workers, logging_steps=cfg.log_every,
        save_strategy="epoch", remove_unused_columns=False, report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=data, data_collator=collate)
    print(f"[train_qwen_lora] {cfg.model_name} LoRA, steps/epoch≈{len(data)//cfg.batch_size}")
    trainer.train()
    fd = os.path.join(cfg.output_dir, "qwen_lora")
    model.save_pretrained(fd); proc.save_pretrained(fd)   # LoRA adapter + processor
    print(f"[train_qwen_lora] done -> {fd}")


def train_la_lora(cfg: TrainConfig, taxonomy: Taxonomy):
    """LoRA fine-tune **LocateAnything** via its autoregressive (AR) path.

    §20 found LA's *PBD* training path blocked (custom block-mask + pos_loss). But its
    `forward(labels=...)` is a plain causal-LM CrossEntropyLoss (modeling:256-266), and
    boxes are *discrete location tokens* `<0>..<1000>` (ids 151677..152677), so it trains
    exactly like Qwen (§21) — just with a location-token target instead of JSON. We freeze
    MoonViT, LoRA the LLM, and supervise the answer:
        <ref>car</ref><box><x1><y1><x2><y2></box>...   (coords = round(c/dim*1000))
    Generate afterwards with generation_mode='slow' (AR, uses the tuned weights). See §22.
    """
    from transformers import AutoModel, AutoTokenizer, AutoProcessor, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model

    name = cfg.model_name or "nvidia/LocateAnything-3B"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(name, trust_remote_code=True)
    model = AutoModel.from_pretrained(name, torch_dtype=torch.bfloat16, trust_remote_code=True)

    # LA's special-token scheme (configuration_locateanything.py) — see §22.1.
    C = model.config
    BOX_S, BOX_E = C.box_start_token_id, C.box_end_token_id           # 151668, 151669
    REF_S, REF_E = C.ref_start_token_id, C.ref_end_token_id           # 151672, 151673
    COORD0 = C.coord_start_token_id                                   # 151677 == '<0>'
    NBINS = C.coord_end_token_id - C.coord_start_token_id             # 1000
    IM_END = getattr(getattr(C, "text_config", None), "eos_token_id", 151645)

    # LoRA on the LLM projections only (MoonViT stays frozen — it is a feature extractor).
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.config.use_cache = False
    # NOTE: no gradient_checkpointing here. LA splices vision embeds with an in-place
    # `input_embeds[selected] = ...` (modeling:240), which conflicts with the
    # enable_input_require_grads() that checkpointing needs. MoonViT is frozen and its
    # 2x2 patch-merge keeps the token count low, so bf16 LoRA at bs=1 fits without it.

    # Work around a bug in LA's *training* path: the inner LLM returns
    # `(output, pos_loss_list)` when `self.training` is True (modeling_qwen2.py:1534),
    # but the outer model calls it without `labels`, so pos_loss_list is unbound AND the
    # outer expects a single `outputs.logits`. The outer already computes a clean CE loss
    # itself, so we force the inner LM's training flag off to take the normal return path.
    # Autograd is unaffected by `.training` — LoRA grads still flow. See TUTORIAL §22.4.
    import types as _types
    _inner = getattr(model.base_model.model, "language_model", None)
    if _inner is not None:
        _orig_fwd = type(_inner).forward
        def _no_mtp_forward(self, *a, _f=_orig_fwd, **k):
            prev = self.training; self.training = False
            try:
                return _f(self, *a, **k)
            finally:
                self.training = prev
        _inner.forward = _types.MethodType(_no_mtp_forward, _inner)

    rep = {uid: terms[0] for uid, terms in taxonomy.open_vocab_terms_by_class().items()}
    image_max = 1024

    def _bin(v, dim):                       # pixel -> location-token id, clamped
        return COORD0 + max(0, min(NBINS, round(float(v) / max(1, dim) * NBINS)))

    def _answer_ids(boxes, labs, W, H):
        """Build the location-token answer: classes grouped, boxes as <box>x1 y1 x2 y2</box>."""
        ids = []
        for uid in sorted(set(int(l) for l in labs)):
            ids += [REF_S] + tok(rep[uid], add_special_tokens=False).input_ids + [REF_E]
            for (x1, y1, x2, y2), l in zip(boxes, labs):
                if int(l) != uid:
                    continue
                ids += [BOX_S, _bin(x1, W), _bin(y1, H), _bin(x2, W), _bin(y2, H), BOX_E]
        return ids + [IM_END]

    data = CanonicalDetData(cfg, taxonomy,
                            split="train" if cfg.dataset == "mixed" else None)

    def collate(samples):
        input_ids, labels_list, pix, grids = [], [], [], []
        for pil, boxes, labs in samples:
            W, H = pil.size                                  # normalize by ORIGINAL size
            img = pil
            if max(W, H) > image_max:                        # downscale only for the encoder
                sc = image_max / max(W, H)
                img = pil.resize((max(1, int(W * sc)), max(1, int(H * sc))))
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Locate all the instances that matches the "
                 "following description: " + ", ".join(rep.values()) + "."}]}]
            text = proc.py_apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids = proc.process_vision_info(msgs)
            enc = proc(text=[text], images=imgs, videos=vids, return_tensors="pt")
            pid = enc["input_ids"][0]
            ans = torch.tensor(_answer_ids(boxes, labs, W, H), dtype=torch.long)
            ids = torch.cat([pid, ans])
            lab = ids.clone(); lab[:pid.shape[0]] = -100     # supervise only the answer
            input_ids.append(ids); labels_list.append(lab)
            pix.append(enc["pixel_values"].to(torch.bfloat16))
            grids.append(torch.as_tensor(enc["image_grid_hws"]))   # LA returns numpy
        maxlen = max(x.shape[0] for x in input_ids)
        pad_id = tok.pad_token_id or 0
        ii = torch.full((len(input_ids), maxlen), pad_id, dtype=torch.long)
        ll = torch.full((len(input_ids), maxlen), -100, dtype=torch.long)
        am = torch.zeros((len(input_ids), maxlen), dtype=torch.long)
        for i, (x, y) in enumerate(zip(input_ids, labels_list)):
            ii[i, :x.shape[0]] = x; ll[i, :y.shape[0]] = y; am[i, :x.shape[0]] = 1
        return {"input_ids": ii, "attention_mask": am, "labels": ll,
                "pixel_values": torch.cat(pix), "image_grid_hws": torch.cat(grids)}

    args = TrainingArguments(
        output_dir=cfg.output_dir, per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=max(1, 8 // cfg.batch_size),
        num_train_epochs=cfg.epochs, learning_rate=cfg.lr, weight_decay=cfg.weight_decay,
        bf16=cfg.amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        dataloader_num_workers=cfg.num_workers, logging_steps=cfg.log_every,
        save_strategy="epoch", remove_unused_columns=False, report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=data, data_collator=collate)
    print(f"[train_la_lora] {name} AR-LoRA, steps/epoch≈{len(data)//cfg.batch_size}")
    trainer.train()
    fd = os.path.join(cfg.output_dir, "la_lora")
    model.save_pretrained(fd); tok.save_pretrained(fd); proc.save_pretrained(fd)
    print(f"[train_la_lora] done -> {fd}")


# ===========================================================================
# YOLO native training (Ultralytics model.train) — proper K-class fine-tuning
# ===========================================================================
def export_to_yolo(data: CanonicalDetData, taxonomy: Taxonomy, out_dir: str,
                   val_frac: float = 0.2) -> str:
    """Export the canonical dataset to **YOLO format on disk** (images + `.txt`
    labels + `data.yaml`) so Ultralytics' native trainer can fine-tune to our K
    unified classes (it resets the head automatically — no COCO-id hack)."""
    n = len(data)
    n_val = max(1, int(n * val_frac))
    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)
    for i in range(n):
        pil, boxes, labels = data[i]
        split = "val" if i < n_val else "train"
        w0, h0 = pil.size
        pil.save(os.path.join(out_dir, "images", split, f"{i:06d}.jpg"))
        lines = []
        for (x1, y1, x2, y2), lab in zip(boxes, labels):
            cx, cy = ((x1 + x2) / 2) / w0, ((y1 + y2) / 2) / h0
            bw, bh = (x2 - x1) / w0, (y2 - y1) / h0
            if bw > 0 and bh > 0:
                lines.append(f"{int(lab)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(os.path.join(out_dir, "labels", split, f"{i:06d}.txt"), "w") as f:
            f.write("\n".join(lines))
    yaml_path = os.path.join(out_dir, "data.yaml")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(taxonomy.classes))
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_dir}\ntrain: images/train\nval: images/val\n"
                f"nc: {taxonomy.num_classes}\nnames:\n{names}\n")
    print(f"[export_to_yolo] wrote {n} imgs ({n_val} val) -> {yaml_path}")
    return yaml_path


def train_yolo_native(cfg: TrainConfig, taxonomy: Taxonomy):
    """Ultralytics native trainer: exports our data to YOLO format, then runs the
    optimized `model.train()` pipeline (mosaic aug, EMA, built-in mAP each epoch)."""
    from ultralytics import YOLO
    data = CanonicalDetData(cfg, taxonomy)
    yolo_data = os.path.join(cfg.output_dir, "yolo_data")
    yaml_path = export_to_yolo(data, taxonomy, yolo_data)
    dev = 0 if cfg.device.startswith("cuda") else "cpu"
    print(f"[train_yolo_native] model={cfg.model_name} (head reset to "
          f"{taxonomy.num_classes} classes) — Ultralytics reports mAP each epoch")
    # Absolute project path so Ultralytics doesn't nest it under its runs/ dir.
    result = YOLO(cfg.model_name).train(
        data=yaml_path, epochs=cfg.epochs, imgsz=cfg.image_size,
        batch=cfg.batch_size, lr0=cfg.lr, device=dev,
        freeze=cfg.freeze,                      # freeze first N layers (anti-forgetting)
        project=os.path.abspath(cfg.output_dir), name="yolo_native", exist_ok=True,
        workers=cfg.num_workers, amp=cfg.amp, verbose=True)
    best = os.path.join(os.path.abspath(cfg.output_dir), "yolo_native", "weights", "best.pt")
    print(f"[train_yolo_native] done -> {best}")
    return best


# ===========================================================================
# CLI
# ===========================================================================
def main():
    cfg = TrainConfig()
    ap = argparse.ArgumentParser(description="ngdet unified detection trainer.")
    ap.add_argument("--trainer", choices=["pytorch", "hf", "gdino", "qwen_lora", "la_lora"],
                    default=cfg.trainer,
                    help="raw loop | HF Trainer (DETR) | gdino | qwen_lora (Qwen2.5-VL LoRA) "
                         "| la_lora (LocateAnything AR-LoRA, §22)")
    ap.add_argument("--backend", choices=["torchvision", "yolo"], default=cfg.backend,
                    help="raw-loop model family (ignored when --trainer hf)")
    ap.add_argument("--yolo-trainer", choices=["raw", "native"], default=cfg.yolo_trainer,
                    help="for --backend yolo: 'raw' (our PyTorch loop) or 'native' "
                         "(Ultralytics model.train, proper K-class fine-tune + built-in mAP)")
    ap.add_argument("--model", dest="model_name", default=cfg.model_name,
                    help="e.g. fasterrcnn_resnet50_fpn_v2 / yolo11n.pt / facebook/detr-resnet-50")
    ap.add_argument("--dataset", default=cfg.dataset)
    ap.add_argument("--root", default=cfg.root)
    ap.add_argument("--taxonomy", default=cfg.taxonomy)
    ap.add_argument("--nuimages-version", default=cfg.nuimages_version)
    ap.add_argument("--eval-version", default=cfg.eval_version,
                    help="nuimages val split for the HF Trainer mAP callback")
    ap.add_argument("--eval-max-images", type=int, default=cfg.eval_max_images)
    ap.add_argument("--max-images", type=int, default=cfg.max_images)
    ap.add_argument("--epochs", type=int, default=cfg.epochs)
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--lr", type=float, default=cfg.lr)
    ap.add_argument("--image-size", type=int, default=cfg.image_size)
    ap.add_argument("--device", default=cfg.device)
    ap.add_argument("--num-workers", type=int, default=cfg.num_workers)
    ap.add_argument("--no-amp", action="store_true")
    # anti-forgetting knobs (§18)
    ap.add_argument("--trainable-backbone-layers", type=int, default=cfg.trainable_backbone_layers,
                    help="torchvision: 0=freeze backbone .. 5=train all (default 3)")
    ap.add_argument("--lr-backbone-mult", type=float, default=cfg.lr_backbone_mult,
                    help="torchvision/HF: backbone LR = lr * this (e.g. 0.1)")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="HF (RT-DETR): freeze the backbone entirely")
    ap.add_argument("--freeze", type=int, default=cfg.freeze,
                    help="YOLO native: freeze the first N layers")
    ap.add_argument("--output-dir", default=cfg.output_dir)
    a = ap.parse_args()

    cfg = TrainConfig(
        trainer=a.trainer, backend=a.backend, yolo_trainer=a.yolo_trainer,
        model_name=a.model_name, dataset=a.dataset, root=a.root, taxonomy=a.taxonomy,
        nuimages_version=a.nuimages_version, eval_version=a.eval_version,
        eval_max_images=a.eval_max_images, max_images=a.max_images,
        epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, image_size=a.image_size,
        device=a.device, num_workers=a.num_workers, amp=not a.no_amp,
        trainable_backbone_layers=a.trainable_backbone_layers,
        lr_backbone_mult=a.lr_backbone_mult, freeze_backbone=a.freeze_backbone,
        freeze=a.freeze, output_dir=a.output_dir)
    taxonomy = Taxonomy.from_preset(cfg.taxonomy)

    if cfg.trainer == "hf":
        train_hf(cfg, taxonomy)
        return
    if cfg.trainer == "gdino":
        train_gdino(cfg, taxonomy)
        return
    if cfg.trainer == "qwen_lora":
        train_qwen_lora(cfg, taxonomy)
        return
    if cfg.trainer == "la_lora":
        train_la_lora(cfg, taxonomy)
        return
    if cfg.backend == "yolo" and cfg.yolo_trainer == "native":
        train_yolo_native(cfg, taxonomy)
        return

    # raw PyTorch loop
    data = CanonicalDetData(cfg, taxonomy)
    backend = build_backend(cfg, taxonomy.num_classes)
    loader = DataLoader(data, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, collate_fn=backend.collate)
    train_loop(cfg, backend, loader)


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
# ===========================================================================
# (1) Raw PyTorch loop, FasterRCNN style, on nuImages real-2D boxes:
#   python -m DeepDataMiningLearning.ngdet.train --trainer pytorch \
#       --backend torchvision --model fasterrcnn_resnet50_fpn_v2 \
#       --dataset nuimages --nuimages-version v1.0-train \
#       --max-images 400 --epochs 5 --batch-size 4
#
# (2) Raw PyTorch loop, YOLO style (Ultralytics model.loss):
#   python -m DeepDataMiningLearning.ngdet.train --trainer pytorch \
#       --backend yolo --model yolo11n.pt --dataset nuimages --max-images 400 \
#       --epochs 5 --batch-size 8 --image-size 640
#
# (3) HuggingFace Trainer, DETR:
#   python -m DeepDataMiningLearning.ngdet.train --trainer hf \
#       --model facebook/detr-resnet-50 --dataset nuimages --max-images 400 \
#       --epochs 5 --batch-size 4
#
# Quick smoke test (tiny): add `--max-images 8 --epochs 1 --batch-size 2`.
# After training, evaluate the checkpoint with the eval harness (run_eval).
# ===========================================================================
if __name__ == "__main__":
    main()
