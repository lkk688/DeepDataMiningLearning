# ngdet Tutorial — Zero-shot 2D detection across datasets

This tutorial walks through using `ngdet` to **pick a detector**, **test it on a
dataset (zero-shot, no training)**, and read the **full COCO evaluation** plus an
optional **human-eval video**. All numbers shown below are from a real run on this
machine (RTX 3090, 50 images/dataset) so you know what to expect.

---

## 0. Environment

Use a Python env that has:

```
torch torchvision transformers ultralytics pycocotools opencv-python matplotlib pyarrow
```

On this machine that is the `py312` conda env:

```bash
PY=/home/lkk688/miniconda/envs/py312/bin/python
```

Always run from the **repo root** (the directory that contains the
`DeepDataMiningLearning` package), so `python -m DeepDataMiningLearning.ngdet.*` resolves.

---

## 1. The mental model

Three pluggable layers (see [README.md](README.md):

```
  detector (any model)  ─┐
                         ├─►  Detection in a UNIFIED taxonomy  ─►  COCO mAP + video
  dataset (any source)  ─┘        (vehicle / person / cyclist)
```

A detector and a dataset speak different "class languages" (COCO's 80 classes vs
KITTI's Car/Pedestrian/Cyclist vs Waymo's Vehicle/...). `ngdet` projects **both**
into one configurable unified taxonomy so any model can be scored on any dataset.

---

## 2. One-command quickstart

```bash
$PY -m DeepDataMiningLearning.ngdet.run_eval \
    --models hf_detr:facebook/detr-resnet-50 yolo:yolo11x.pt \
    --datasets coco kitti nuimages \
    --max-images 50 --score-thr 0.3 --device cuda --video --video-max 40 \
    --coco-ann /mnt/e/Shared/Dataset/coco2017/annotations/instances_val2017.json
```

This evaluates **2 models × 3 datasets = 6 runs** and writes a heatmap, a Markdown
report (with full COCO tables), per-pair videos, and `metrics.json` to
`ngdet/output/` (git-ignored). First run downloads the model weights.

**Full reproduction commands** (the exact runs behind §12's results) are in §12.

---

## 3. Picking a MODEL — the `--models` spec

Each model is `backend:checkpoint`. Pass as many as you want (space-separated).

| Backend key | Adapter | Example specs | Notes |
|---|---|---|---|
| `torchvision` | `detectors/torchvision_det.py` | `torchvision:fasterrcnn_resnet50_fpn_v2`, `torchvision:retinanet_resnet50_fpn_v2`, `torchvision:fcos_resnet50_fpn` | classic CNN baselines |
| `hf_detr` | `detectors/hf_detr.py` | `hf_detr:facebook/detr-resnet-50`, `hf_detr:PekingU/rtdetr_r50vd`, `hf_detr:SenseTime/deformable-detr` | any HF `AutoModelForObjectDetection` |
| `yolo` | `detectors/yolo.py` | `yolo:yolov8x.pt`, `yolo:yolo11x.pt`, `yolo:yolo26x.pt`, `yolo:yolov8s-worldv2.pt` | Ultralytics; `-world` = open-vocab |
| `gdino` | `detectors/grounding_dino.py` | `gdino:IDEA-Research/grounding-dino-tiny`, `gdino:IDEA-Research/grounding-dino-base` | open-vocab, text-prompted |
| `owlv2` | `detectors/grounding_dino.py` | `owlv2:google/owlv2-base-patch16-ensemble` | open-vocab (OWL family) |
| `locate_anything` | `detectors/locate_anything.py` | `locate_anything:nvidia/LocateAnything-3B` | VLM, open-vocab (experimental) |

Closed-set models (torchvision/DETR/YOLO) are mapped to the unified taxonomy by class
**name**. Open-vocab models (Grounding DINO, OWLv2, YOLO-World, LocateAnything) are
**prompted** with the unified class names, so they need no mapping table.

> **Open-vocab caveat (important for fair comparison):** `gdino`/`owlv2` are highly
> *prompt-* and *score-scale-sensitive*. A generic prompt like `"vehicle"` works far
> worse than specific words (`"car . truck . bus"`), and OWL confidences sit on a
> lower scale, so a fixed `--score-thr 0.3` under-counts them. Their lower mAP in the
> heatmap reflects prompt/threshold mismatch — NOT that open-vocab detection is worse.
> Tune the prompt synonyms (in `taxonomy.py`) and threshold per family before concluding.

---

## 4. Picking a DATASET — `--datasets` and `--roots`

Supported: `coco kitti waymo nuscenes nuimages`. Default roots are in
`run_eval.py::DEFAULT_ROOTS`; override any with `--roots name=path`.

- **coco** — needs `--coco-ann /path/to/instances_val2017.json`; root = the
  `val2017/` image dir. This is the *same-domain reference* for the COCO-pretrained models.
- **kitti** — root = KITTI dir with `training/image_2` + `label_2`. Native 2D boxes.
- **waymo** — root = the Waymo v2 parquet dir. Native 2D boxes.
- **nuscenes** — root must contain a `v1.0-trainval` subfolder of JSONs. The full
  trainval JSONs are huge (1.3 GB); for quick runs use the **mini** split via a
  symlink root:

  ```python
  from DeepDataMiningLearning.detection.verify_datasets_video import build_nuscenes_mini_root
  build_nuscenes_mini_root("/mnt/e/Shared/Dataset/NuScenes/v1.0-mini",
                           "DeepDataMiningLearning/ngdet/output/nuscenes_mini_root")
  ```
  then `--roots nuscenes=DeepDataMiningLearning/ngdet/output/nuscenes_mini_root`.
- **nuimages** — root = an extracted nuImages dir containing `samples/` + a
  `v1.0-mini` (or `v1.0-train`/`v1.0-val`) folder of JSONs. nuImages has **real
  human-annotated 2D boxes** (`object_ann.json`), so it is the RELIABLE 2D driving
  benchmark (use it instead of `nuscenes`, whose 2D boxes are projected from 3D and
  are loose — see §8). Extract the mini split with:
  `tar xzf nuimages-v1.0-mini.tgz -C <dir>/nuimages`.

`--max-images N` caps each dataset (use 0 for all). Keep `--score-thr` **identical**
across models, otherwise the comparison is not apples-to-apples.

---

## 5. What you get — the outputs (`ngdet/output/`)

| File | What it is |
|---|---|
| `metrics.json` | machine-readable metrics for every run (full 12 COCO stats + per-class) |
| `report.md` | the human report: summary matrix + generalization gap + **full COCO tables** |
| `heatmap_mAP.png` | the model × dataset generalization heatmap |
| `video_<model>__<dataset>.mp4` | annotated clip (pred = colored, GT = green) |
| `pred_<model>__<dataset>.json` | per-image pred/GT counts for analysis |

### 5a. The generalization heatmap (10 models × 5 datasets)

![heatmap](docs/heatmap_mAP.png)

Rows = models (grouped: classic CNN → DETR family → YOLO → open-vocab → VLM),
columns = test domain, cell = mAP. Full numbers + reproduction commands are in §12.
Four things to read off it:

1. **In-domain COCO is brightest** (left column, ~0.5–0.6) — these are all
   COCO-pretrained, so COCO is their home turf.
2. **Cross-domain compresses the field** — on KITTI every model lands ~0.23–0.28
   regardless of architecture. *Domain shift hurts more than model choice*: the
   2017 Faster R-CNN (0.279) ties the 2025 YOLO26x (0.264). This is the single most
   important teaching point — picking a fancier detector buys little zero-shot
   generalization; adapting to the domain is what matters.
3. **The nuscenes column is dark, the nuimages column is bright** — same scenes,
   but nuscenes 2D GT is *projected from 3D* (loose) while nuimages is *real 2D*.
   The ≈5× jump (§12.2) is a GT-quality artifact, not a model effect.
4. **Open-vocab (GDINO/OWLv2) is competitive and leads on nuImages** — after the
   prompt fix (§3 caveat), text-prompted detectors generalize at least as well as
   the closed-set COCO detectors. The **VLM row (LocateAnything) looks weak only
   because mAP can't rank its unscored boxes** (§12.1 †).

### 5b. The summary matrix (from `report.md`)

`report.md` also prints a model×dataset matrix and a **generalization-gap** table
(`COCO mAP − driving-domain mAP`, higher = worse transfer):

```
Generalization gap        kitti    waymo    nuscenes  nuimages
Faster R-CNN v2          +0.302   +0.258   +0.519    +0.282
YOLO26x                  +0.343   +0.361   +0.541    +0.348
```

→ YOLO's gap is *larger* than Faster R-CNN's: it wins in-domain but transfers worse.

---

## 6. The FULL COCO evaluation table (not just mAP)

For **every** run, `report.md` contains the complete canonical 12-row COCO table
(6 AP + 6 AR) plus per-class AP — identical to what `pycocotools` prints. Example
(`hf_detr:facebook/detr-resnet-50 × coco`):

| Metric | Value |
|---|---|
| AP @[IoU=0.50:0.95 \| area=all \| maxDets=100] | 0.376 |
| AP @[IoU=0.50 \| area=all \| maxDets=100]      | 0.530 |
| AP @[IoU=0.75 \| area=all \| maxDets=100]      | 0.389 |
| AP @[IoU=0.50:0.95 \| area=small \| maxDets=100]  | 0.275 |
| AP @[IoU=0.50:0.95 \| area=medium \| maxDets=100] | 0.415 |
| AP @[IoU=0.50:0.95 \| area=large \| maxDets=100]  | 0.770 |
| AR @[IoU=0.50:0.95 \| area=all \| maxDets=1]   | 0.134 |
| AR @[IoU=0.50:0.95 \| area=all \| maxDets=10]  | ...   |
| AR @[IoU=0.50:0.95 \| area=all \| maxDets=100] | ...   |
| AR small / medium / large                       | ...   |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.574 |
| person  | 0.553 |
| cyclist | 0.000 |

**How to read it:** `AP@[.5:.95]` is the headline mAP (averaged over 10 IoU
thresholds). `AP50`/`AP75` are loose/strict localization. `small/medium/large`
split by object area — note detectors are much stronger on `large` (0.770) than
`small` (0.275). `AR` rows are recall ceilings at 1/10/100 detections per image.

You can also regenerate the report/heatmap from a saved `metrics.json` without
re-running models:

```bash
$PY -m DeepDataMiningLearning.ngdet.report --metrics DeepDataMiningLearning/ngdet/output/metrics.json
```

---

## 7. Video for human eval (`--video`)

- `--video` writes one mp4 per (model, dataset): **predictions in per-class colors,
  ground truth in green**, with a `pred=color GT=green` banner.
- `--video-max N` writes only the first N frames to each clip (a quick human-eval
  sample) while **all** images still count toward mAP. Great for eyeballing failure
  modes without a huge file.

---

## 8. Interpreting results — important caveats

These are *features* of the harness surfacing real issues, not bugs:

1. **NuScenes mAP is low (~0.06) — use nuImages instead.** NuScenes has no native
   2D boxes; its GT here is **3D boxes projected to 2D** (axis-aligned envelope),
   looser than the true visible extent, so IoU with tight predictions is low.
   **nuImages** has real human-annotated 2D boxes: the SAME models score **~0.31–0.37
   mAP on nuImages vs ~0.06 on projected nuscenes (≈5× higher)** — proof the gap was
   the GT, not the models. Keep both columns for the contrast; trust nuImages.

2. **`cyclist` AP is ~0 for COCO models.** COCO annotates `person` + `bicycle`
   separately, while KITTI/Waymo annotate a combined `Cyclist`. The taxonomy folds
   COCO `bicycle`→`cyclist`, but the box geometry differs, so IoU rarely passes.
   This is a documented taxonomy-mismatch finding (see `taxonomy.py`).

3. **Waymo `person` = 0 at 50 images.** Small-sample artifact (the first 50
   time-ordered frames are mostly vehicle-heavy / few clear pedestrians). Increase
   `--max-images` for a stable number.

4. **Keep `--score-thr` fixed** across models or mAP rankings are not comparable.

---

## 9. Changing the unified taxonomy

The label space is configurable via `--taxonomy`:

```
--taxonomy driving3        # vehicle / person / cyclist   (default)
--taxonomy vehicle_person  # 2-class (the categories every AD dataset agrees on)
--taxonomy driving5        # car / truck / bus / person / cyclist
```

Add your own in `taxonomy.py::TAXONOMY_PRESETS` as `{unified_name: [synonyms]}`.

---

## 10. Add your own model (≈50 lines)

```python
# DeepDataMiningLearning/ngdet/detectors/my_backend.py
from .base import BaseDetector, Detection, register

@register("my_backend")
class MyDetector(BaseDetector):
    def __init__(self, taxonomy, device="cuda", score_thr=0.3, model_name=None, **kw):
        super().__init__(taxonomy, device, score_thr)
        # load model; for a closed-set model build the native->unified LUT:
        #   self.id_lut = taxonomy.build_id_lut({id: name, ...})
    def predict(self, image, prompt=None) -> Detection:
        # run model -> native xyxy boxes/scores/ids, then:
        return self._fold_to_unified(boxes_xyxy, scores, native_ids)
```

Import it in `detectors/base.py::build_detector` (next to the others) and run with
`--models my_backend:checkpoint`.

---

## 11. Latency / throughput benchmark (native vs accelerated)

`ngdet.latency` times end-to-end `predict()` (with CUDA sync) and compares the
native PyTorch/HF/Ultralytics path against accelerated backends. Modes a model
can't do are skipped.

```bash
python -m DeepDataMiningLearning.ngdet.latency \
    --models yolo:yolo11x.pt yolo:yolov8x.pt \
             hf_detr:facebook/detr-resnet-50 hf_detr:PekingU/rtdetr_r50vd \
             torchvision:fasterrcnn_resnet50_fpn_v2 \
    --accel fp32 fp16 compile tensorrt --iters 40
```

Writes `latency_report.md` (mean/p50/p90 ms, FPS, speedup), `latency_fps.png`
(grouped bar), and `latency.json` to `ngdet/output/`.

![latency](docs/latency_fps.png)

**Accel support per family** (unsupported pairs auto-skip):
- **FP16** — yolo / hf_detr / torchvision
- **torch.compile** — hf_detr / torchvision
- **TensorRT, ONNX** — yolo (Ultralytics native `.engine`/`.onnx`) AND hf_detr
  (DETR via ONNX export + **onnxruntime-gpu** CUDA/TensorRT Execution Providers —
  set `LD_LIBRARY_PATH` to torch's bundled cuDNN, see §11 note in `latency.py`).
- Not wired: torch-tensorrt / OpenVINO (not installed); vLLM serving for the
  LocateAnything VLM (vLLM lacks this custom arch). RT-DETR's deformable-attention
  ops do not export to ONNX/TRT, so its onnx/tensorrt rows auto-skip.

**Measured on an RTX 3090, what to expect:**

| family | native FPS | best accel | speedup |
|---|---|---|---|
| YOLO11x / YOLOv8x | 57 / 61 | **TensorRT** | **×1.32 / ×1.24** (→76 FPS) |
| Faster R-CNN v2 | 28 | FP16 | ×1.28 |
| DETR-R50 | 21 | **compile ×1.82**, TensorRT ×1.25, ONNX ×1.05 | |
| RT-DETR-R50 | 20 | **torch.compile ×2.16** | |

Teaching point: the best accelerator is **architecture-dependent**. YOLO loves
**TensorRT** (NMS-free graph maps cleanly, ×1.3). The DETR family barely benefits
from FP16 (×0.96) but gains most from **torch.compile** (fusing attention ops) —
DETR ×1.82, RT-DETR ×2.16; TensorRT helps DETR too (×1.25) but less than compile.
The CNN-based Faster R-CNN is the opposite — **FP16** helps most (×1.28).

## 11b. Per-module self-tests (no full pipeline)

```bash
$PY -m DeepDataMiningLearning.ngdet.taxonomy            # label folding demo
$PY -m DeepDataMiningLearning.ngdet.datasets --name kitti --max-images 3
$PY -m DeepDataMiningLearning.ngdet.evaluator           # synthetic mAP~1.0 sanity
$PY -m DeepDataMiningLearning.ngdet.video               # 3-frame KITTI GT clip
$PY -m DeepDataMiningLearning.ngdet.detectors.hf_detr   # DETR on a COCO sample
$PY -m DeepDataMiningLearning.ngdet.detectors.yolo      # YOLO on a sample image
```

---

## 12. Full reference results + exact reproduction commands

All numbers below were produced on an **RTX 3090** with the commands shown. `$PY` is
the env from §0; first runs download model weights.

### 12.1 Accuracy — zero-shot mAP@[.5:.95], 10 models × 5 datasets

| model | coco | kitti | waymo | nuscenes | nuimages |
|---|---|---|---|---|---|
| torchvision: Faster R-CNN v2 | 0.581 | 0.279 | 0.323 | 0.062 | 0.299 |
| torchvision: RetinaNet v2 | 0.512 | 0.267 | 0.272 | 0.077 | 0.265 |
| hf_detr: DETR-R50 | 0.473 | 0.231 | 0.226 | 0.054 | 0.223 |
| hf_detr: RT-DETR-R50 | 0.596 | 0.262 | 0.269 | 0.068 | 0.256 |
| yolo: YOLOv8x | 0.591 | 0.255 | 0.252 | 0.060 | 0.244 |
| yolo: YOLO11x | 0.573 | 0.260 | 0.249 | 0.065 | 0.249 |
| yolo: YOLO26x | **0.607** | 0.264 | 0.246 | 0.066 | 0.259 |
| gdino: Grounding-DINO-tiny | 0.527 | 0.223 | 0.311 | 0.063 | **0.306** |
| owlv2: OWLv2-base | 0.533 | 0.191 | 0.292 | 0.073 | 0.277 |
| locate_anything: LocateAnything-3B† | 0.146 | 0.058 | 0.132 | 0.004 | 0.055 |

Samples: coco/kitti/waymo/nuscenes = 120 imgs; **nuimages = 500 (v1.0-val)**; LA = 30.
Waymo used `--waymo-stride 40`; open-vocab used `gdino@0.15`, `owlv2@0.05`.

† **LocateAnything mAP is structurally understated** — the VLM emits no calibrated
confidence (all boxes score 1.0), and mAP needs ranked detections, so AP is
penalized even though the boxes are visually correct. Read its row as "runs &
localizes", not as a quality ranking.

**Headline reads:** in-domain COCO leaders are YOLO26x / RT-DETR (~0.60); cross-domain
**everything compresses to ~0.25 on KITTI** regardless of architecture (domain shift
dominates model choice); open-vocab (GDINO/OWLv2) is competitive after the prompt fix
and **leads on nuImages**.

### 12.2 nuScenes (projected 3D) vs nuImages (real 2D) — same models

| | nuscenes (projected) | nuimages (real 2D) | ratio |
|---|---|---|---|
| Faster R-CNN | 0.062 | 0.299 | **4.8×** |
| Grounding DINO | 0.063 | 0.306 | **4.9×** |
| YOLO26x | 0.066 | 0.259 | 3.9× |

→ nuScenes' 2D GT is **projected from 3D (loose envelope)**, so low mAP there is a
**GT artifact**, not model failure. **Use nuImages for 2D driving eval.**

### 12.3 Latency — native vs accelerated (FPS, RTX 3090)

| model | fp32 | fp16 | compile | tensorrt | onnx |
|---|---|---|---|---|---|
| YOLO11x | 57 | 52 | — | **76 (×1.32)** | 36 |
| YOLOv8x | 61 | 71 | — | **75 (×1.24)** | 33 |
| Faster R-CNN v2 | 28 | **36 (×1.28)** | 31 | — | — |
| DETR-R50 | 21 | 20 | **38 (×1.82)** | 26 (×1.25) | 22 |
| RT-DETR-R50 | 20 | 19 | **43 (×2.16)** | skip‡ | skip‡ |

‡ RT-DETR's deformable-attention ops don't export to ONNX/TensorRT (auto-skipped).
**Best accelerator is architecture-dependent**: YOLO→TensorRT, DETR-family→torch.compile,
CNN R-CNN→FP16.

### 12.4 Exact commands

```bash
# --- (A) Accuracy: 9 models × {coco,kitti,waymo,nuscenes} ---
$PY -m DeepDataMiningLearning.ngdet.run_eval \
  --models torchvision:fasterrcnn_resnet50_fpn_v2 torchvision:retinanet_resnet50_fpn_v2 \
           hf_detr:facebook/detr-resnet-50 hf_detr:PekingU/rtdetr_r50vd \
           yolo:yolov8x.pt yolo:yolo11x.pt yolo:yolo26x.pt \
           gdino:IDEA-Research/grounding-dino-tiny@0.15 \
           owlv2:google/owlv2-base-patch16-ensemble@0.05 \
  --datasets coco kitti waymo nuscenes --max-images 120 --waymo-stride 40 \
  --device cuda --video --video-max 16 \
  --coco-ann <coco>/annotations/instances_val2017.json \
  --roots nuscenes=DeepDataMiningLearning/ngdet/output/nuscenes_mini_root

# --- (B) nuImages val (real 2D), large sample ---
$PY -m DeepDataMiningLearning.ngdet.run_eval \
  --models <same 9 models> \
  --datasets nuimages --nuimages-version v1.0-val --max-images 500 \
  --device cuda --out-dir DeepDataMiningLearning/ngdet/output/nuimages_run

# --- (C) LocateAnything VLM row (slow; smaller sample) ---
$PY -m DeepDataMiningLearning.ngdet.run_eval \
  --models locate_anything:nvidia/LocateAnything-3B \
  --datasets coco kitti waymo nuscenes --max-images 30 --waymo-stride 40 \
  --device cuda --out-dir DeepDataMiningLearning/ngdet/output/locate_anything

# --- (D) Latency (DETR onnx/tensorrt need the LD_LIBRARY_PATH from README §4) ---
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$(ls -d $SP/nvidia/*/lib | tr '\n' ':')$LD_LIBRARY_PATH
$PY -m DeepDataMiningLearning.ngdet.latency \
  --models yolo:yolo11x.pt yolo:yolov8x.pt \
           hf_detr:facebook/detr-resnet-50 hf_detr:PekingU/rtdetr_r50vd \
           torchvision:fasterrcnn_resnet50_fpn_v2 \
  --accel fp32 fp16 compile tensorrt onnx --iters 40
```

Runs (A)/(B)/(C) write separate `metrics.json` files; combine them into one heatmap
by concatenating the metrics lists and calling `report.write_markdown_report` /
`report.plot_heatmap` (or just run `report.py --metrics <file>` per run).

---

## 13. Datasets in depth — native format → unified format

Every dataset speaks a different format. The job of [datasets.py](datasets.py) is to
convert each into one common `EvalSample(image, gt_boxes [xyxy px], gt_labels [unified id])`.
Below: what the raw data looks like, and where the conversion lives.

### 13.1 COCO — `instances_val2017.json`
- **Native**: one big JSON with `images`, `annotations` (`bbox = [x, y, w, h]`,
  `category_id`), `categories`. Box = top-left + width/height.
- **Convert**: `torchvision.datasets.CocoDetection` reads it; we do xywh→xyxy and map
  `category_id`→unified by the category *name*.
- **Code**: [`EvalDataset._build_coco`](datasets.py) · ref: [cocodataset.org/#format-data](https://cocodataset.org/#format-data)

### 13.2 KITTI — per-image label `.txt`
- **Native**: `training/label_2/000000.txt`, one object per line:
  `type truncation occlusion alpha x1 y1 x2 y2 h w l X Y Z ry`. Columns 4–7 are the
  **2D box (xyxy pixels)** — already native 2D.
- **Convert**: reuse `detection/dataset_kitti.py::KittiDataset` (returns torch
  `{boxes, labels}`); map KITTI ids (`Car/Pedestrian/Cyclist/...`) → unified.
- **Code**: [`_build_kitti`](datasets.py), [dataset_kitti.py](../detection/dataset_kitti.py) ·
  ref: [KITTI object benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php)

### 13.3 Waymo Open (v2) — Parquet
- **Native**: column-store parquet. `camera_image/*.parquet` (JPEG bytes per frame),
  `camera_box/*.parquet` with `[CameraBoxComponent].box.center.{x,y}` +
  `size.{x,y}` (center+size **pixels**) and `.type` (1=Vehicle,2=Pedestrian,3=Cyclist).
  Frames are grouped **by 20 s segment**.
- **Convert**: `detection/dataset_waymov3_1.py::Waymo2DDataset` reads parquet,
  cx,cy,w,h→xyxy; **`--waymo-stride` is needed** so a sample spans many segments
  (a contiguous slice = the first vehicle-heavy segment only).
- **Code**: [`_build_waymo`](datasets.py), [dataset_waymov3_1.py](../detection/dataset_waymov3_1.py) ·
  ref: [waymo.com/open](https://waymo.com/open/) · [format docs](https://github.com/waymo-research/waymo-open-dataset)

### 13.4 nuScenes — JSON tables, **3D→2D projection**
- **Native**: relational JSON (`sample_data`, `sample_annotation` with 3D box
  `translation/size/rotation`, `calibrated_sensor`, `ego_pose`). **No native 2D boxes.**
- **Convert**: `detection/dataset_nuscenes.py` projects the 3D cuboid's 8 corners to the
  camera and takes the **axis-aligned min/max envelope** → a *loose* 2D box. We added
  `min_visibility`/`require_lidar_pts` filtering, but looseness is inherent (§8).
- **Code**: [`_build_nuscenes`](datasets.py), [dataset_nuscenes.py](../detection/dataset_nuscenes.py) ·
  ref: [nuscenes.org](https://www.nuscenes.org/nuscenes)

### 13.5 nuImages — JSON, **real human-annotated 2D** (recommended)
- **Native**: `object_ann.json` has `bbox = [x1, y1, x2, y2]` (pixels, real 2D) +
  `category_token` + optional instance `mask`; `sample_data.json` lists key-frame images.
- **Convert**: we parse the JSON directly (no devkit dependency), group annotations by
  image, and map the dotted category (`vehicle.bicycle`, `human.pedestrian.adult`) to
  unified by **right-to-left token matching** (`bicycle`→cyclist before `vehicle`).
- **Code**: [`_build_nuimages` + `nuimages_name_to_unified`](datasets.py) ·
  ref: [nuimages.org](https://www.nuscenes.org/nuimages)

> **Label unification** for all of the above lives in [taxonomy.py](taxonomy.py):
> `Taxonomy.build_id_lut` maps a source `{id: name}` table to unified ids by name, so
> a class outside the active taxonomy (e.g. `traffic_cone`) folds to "ignore".

---

## 14. Model architectures — structure, loss, training, code

For each family below: **how it's built**, the **loss function (with math)**, the
**training recipe**, and links to the **reference implementation** (the actual model
code you can read) plus our thin **adapter** in [detectors/](detectors/).
Math renders on GitHub (LaTeX). Notation: $b=(x,y,w,h)$ a box, $p$ a class
probability, $\mathbb{1}[\cdot]$ an indicator, $N_{pos}$ the number of positives.

---

### 14.1 Two-stage CNN — Faster R-CNN

**Structure.** Backbone (ResNet-50 + FPN) → **Region Proposal Network (RPN)** slides
over feature maps and, per anchor, predicts objectness + a box delta → top proposals
go through **RoIAlign** → a **detection head** outputs per-class scores + refined
boxes. Anchor-based; needs NMS at inference.

**Loss.** A multi-task sum, applied at both the RPN and the head:

$$ L(\{p_i\},\{t_i\}) = \frac{1}{N_{cls}}\sum_i L_{cls}(p_i,p_i^*) \;+\; \lambda\,\frac{1}{N_{reg}}\sum_i p_i^*\,L_{reg}(t_i,t_i^*) $$

- $L_{cls}$ = log loss (RPN: object/background; head: softmax over classes).
- $L_{reg}$ = smooth-L1 on **parameterized** deltas $t_x=(x-x_a)/w_a,\; t_w=\log(w/w_a)$
  (subscript $a$ = anchor); only positives ($p_i^*=1$) contribute.
- $\text{smooth}_{L1}(x)=0.5x^2$ if $|x|<1$, else $|x|-0.5$.

**Training.** Anchors labeled by IoU (≥0.7 → positive, <0.3 → negative); RPN +
detector trained **jointly** with SGD; balanced minibatch of anchors/RoIs.

**Code.** torchvision: [faster_rcnn.py](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py),
[rpn.py](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/rpn.py),
[roi_heads.py](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py)
· paper [Faster R-CNN (2015)](https://arxiv.org/abs/1506.01497) · adapter [torchvision_det.py](detectors/torchvision_det.py)

---

### 14.2 One-stage CNN — RetinaNet (anchor) & FCOS (anchor-free)

**Structure.** Dense per-location prediction in one pass. RetinaNet: FPN + two small
subnets (cls, box) over anchors. FCOS: FPN + per-pixel heads, **no anchors** — each
foreground pixel regresses 4 distances $(l,t,r,b)$ to the box sides + a centerness.

**Loss — RetinaNet** solves foreground/background imbalance with **focal loss**:

$$ FL(p_t) = -\alpha_t\,(1-p_t)^{\gamma}\,\log(p_t), \qquad \gamma=2,\ \alpha=0.25 $$

where $p_t=p$ if the anchor is positive else $1-p$. Box term = smooth-L1 on anchor deltas.

**Loss — FCOS**: focal (cls) + IoU/GIoU (box) + BCE (centerness):

$$ L = \tfrac{1}{N_{pos}}\!\sum L_{cls} + \tfrac{\lambda}{N_{pos}}\!\sum \mathbb{1}[c^*>0]\,L_{reg} + \tfrac{1}{N_{pos}}\!\sum \mathbb{1}[c^*>0]\,L_{ctr},\quad
\text{ctr}^*=\sqrt{\tfrac{\min(l,r)}{\max(l,r)}\cdot\tfrac{\min(t,b)}{\max(t,b)}} $$

Centerness down-weights boxes predicted far from object centers (suppresses low-quality far predictions).

**Training.** RetinaNet: every anchor contributes to focal loss (no sampling needed).
FCOS: a pixel is positive if it falls in a GT box (+ FPN scale range); centerness
multiplies the score at test time.

**Code.** torchvision: [retinanet.py](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py),
[fcos.py](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/fcos.py)
· papers [Focal Loss/RetinaNet (2017)](https://arxiv.org/abs/1708.02002),
[FCOS (2019)](https://arxiv.org/abs/1904.01355) · adapter [torchvision_det.py](detectors/torchvision_det.py)

---

### 14.3 Transformer set-prediction — DETR & RT-DETR

**Structure.** CNN backbone → transformer **encoder** (over flattened features) →
**decoder** with $N$ learned **object queries** → each query → (class, box). **No
anchors, no NMS.** RT-DETR replaces the heavy encoder with an efficient hybrid
CNN+attention encoder and adds IoU-aware query selection for real-time speed.

**Loss.** First find the optimal one-to-one assignment $\hat\sigma$ between the $N$
predictions and the (padded) GT set via the **Hungarian algorithm**:

$$ \hat\sigma=\arg\min_{\sigma}\sum_{i}\Big[-\mathbb{1}[c_i\neq\varnothing]\,\hat p_{\sigma(i)}(c_i) + \mathbb{1}[c_i\neq\varnothing]\,L_{box}(b_i,\hat b_{\sigma(i)})\Big] $$

then minimize the **Hungarian loss** over that matching:

$$ \mathcal{L}=\sum_{i}\Big[-\log\hat p_{\hat\sigma(i)}(c_i) + \mathbb{1}[c_i\neq\varnothing]\,L_{box}(b_i,\hat b_{\hat\sigma(i)})\Big],\quad
L_{box}=\lambda_{L1}\lVert b-\hat b\rVert_1 + \lambda_{iou}\,L_{GIoU} $$

(RT-DETR swaps the cls term for **varifocal loss** and adds denoising queries.)

**Training.** AdamW, long schedule (original DETR: 300–500 epochs), auxiliary losses
at every decoder layer; no NMS/anchor hyper-params to tune. The bipartite matching is
what removes duplicate predictions (one query per object).

**Code.** transformers: [modeling_detr.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/detr/modeling_detr.py)
(`DetrHungarianMatcher`, `DetrLoss`),
[modeling_rt_detr.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/rt_detr/modeling_rt_detr.py)
· papers [DETR (2020)](https://arxiv.org/abs/2005.12872), [RT-DETR (2023)](https://arxiv.org/abs/2304.08069)
· adapter [hf_detr.py](detectors/hf_detr.py)

---

### 14.4 Real-time one-stage — YOLOv8 / YOLO11 / YOLO26

**Structure.** Backbone (CSP) + neck (PAN-FPN) + **decoupled, anchor-free head**
(v8+). The box branch predicts each side as a **discrete distribution over bins**
(Distribution Focal Loss). v10/YOLO26 add **NMS-free** inference via consistent
dual one-to-one + one-to-many assignment.

**Loss.** Three terms, with a **task-aligned assigner** choosing positives:

$$ L = \lambda_{box}\,L_{CIoU} + \lambda_{dfl}\,L_{DFL} + \lambda_{cls}\,L_{BCE} $$

- **CIoU** (box): $\;L_{CIoU}=1-IoU+\dfrac{\rho^2(b,b^{gt})}{c^2}+\alpha v\;$ — adds
  center-distance ($\rho$, $c$ = diagonal of the enclosing box) and aspect-ratio ($v$) penalties.
- **DFL** (box sharpness): treats an edge offset $y$ between bins $y_i,y_{i+1}$ as a
  distribution, $\;L_{DFL}=-\big[(y_{i+1}-y)\log S_i+(y-y_i)\log S_{i+1}\big]$.
- **BCE** (class): binary cross-entropy per class (multi-label friendly).
- **Assigner (TaskAligned):** picks positives by a score $t=p^{\alpha}\cdot u^{\beta}$
  (classification $p$ × IoU $u$).

**Training.** Heavy augmentation (mosaic, mixup, copy-paste), EMA weights, cosine LR;
mosaic turned off for the last epochs. NMS at inference (except NMS-free v10/26).

**Code.** ultralytics: [head.py `Detect`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py),
[loss.py `v8DetectionLoss`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py),
[tal.py `TaskAlignedAssigner`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tal.py)
· refs [Ultralytics docs](https://docs.ultralytics.com/models/), [YOLOv10 NMS-free (2024)](https://arxiv.org/abs/2405.14458)
· adapter [yolo.py](detectors/yolo.py)

---

### 14.5 Open-vocabulary — Grounding DINO & OWLv2

**Structure.** Detect *arbitrary* text-named classes. **Grounding DINO** = a DINO
(DETR-style) detector + a BERT text encoder + a **feature enhancer** that does
image↔text cross-attention; queries are aligned to text tokens. **OWLv2** = a CLIP
ViT where each patch/output token emits a box + an embedding, scored by dot-product
against the **text query embeddings**.

**Loss.** Same set-prediction backbone as DETR (Hungarian matching + L1 + GIoU), but
the classification cost becomes a **contrastive / region-text alignment** term — a
query is "class $k$" if its embedding $q$ aligns with the text embedding $t_k$:

$$ s_{k}=\langle q,\,t_k\rangle,\qquad L_{align}=\text{focal/BCE}\big(\sigma(s_k),\,y_k\big) $$

OWLv2 additionally **self-trains** on pseudo-box labels mined from web image-text data.

**Training.** Pretrain on huge grounded/detection+caption corpora (Objects365, GoldG,
…); at inference you just pass class names as the prompt — no fine-tuning needed
(that's why ngdet uses them zero-shot).

**Code.** transformers: [modeling_grounding_dino.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/grounding_dino/modeling_grounding_dino.py),
[modeling_owlv2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlv2/modeling_owlv2.py)
· original [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
· papers [Grounding DINO (2023)](https://arxiv.org/abs/2303.05499), [OWLv2 (2023)](https://arxiv.org/abs/2306.09683)
· adapter [grounding_dino.py](detectors/grounding_dino.py)

---

### 14.6 VLM grounding — NVIDIA LocateAnything-3B

**Structure.** A **vision-language model** (Moon-ViT vision encoder + Qwen2.5 LLM
decoder + MLP projector) that **generates boxes as text**. Output interleaves a class
tag and its boxes: `<ref>car</ref><box><x1><y1><x2><y2></box>...` with coords in
[0,1000]. **Parallel Box Decoding (PBD)** emits all 4 coords of a box as one atomic
block (the fast "MTP" mode) instead of one coordinate token at a time.

**Loss.** Trained like an LLM — **autoregressive cross-entropy** over the output token
sequence (the special box/coordinate tokens included):

$$ \mathcal{L} = -\sum_{t}\log P_\theta\big(x_t \mid x_{<t},\,\text{image}\big) $$

plus the PBD head that predicts the coordinate set in parallel. There is **no
explicit IoU/box-regression loss and no confidence score** — which is exactly why its
mAP here is structurally understated (§12.1 †): mAP needs ranked detections.

**Training.** Vision-language pretraining + instruction tuning on a large grounding
dataset (138 M queries, 785 M boxes per the paper); detection becomes one of several
localization "skills" the VLM is taught.

**Code.** model repo (custom `modeling_locateanything.py`, `trust_remote_code`):
[nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B)
· paper [LocateAnything (2026)](https://arxiv.org/abs/2605.27365) · adapter [locate_anything.py](detectors/locate_anything.py)

---

### 14.7 At a glance

| family | anchors? | NMS? | classification target | box loss | matching/assigner |
|---|---|---|---|---|---|
| Faster R-CNN | ✅ | ✅ | softmax | smooth-L1 | IoU to anchors |
| RetinaNet | ✅ | ✅ | focal | smooth-L1 | IoU to anchors |
| FCOS | ❌ | ✅ | focal + centerness | GIoU | in-box + scale |
| DETR / RT-DETR | ❌ | ❌ | CE / varifocal | L1 + GIoU | Hungarian (1-to-1) |
| YOLOv8/11/26 | ❌ | ✅/❌¹ | BCE | CIoU + DFL | TaskAligned |
| GDINO / OWLv2 | ❌ | ❌/✅ | region-text contrastive | L1 + GIoU | Hungarian |
| LocateAnything | ❌ | ❌ | token cross-entropy (LM) | — (generated) | — |

¹ YOLO26 / v10 are NMS-free.

---

## 15. Acceleration in depth — how the speedups work

The [latency.py](latency.py) benchmark applies these to the same model and times
end-to-end `predict()`. Each adapter implements the modes it supports (others skip).

### 15.1 FP16 (half precision)
Run weights/activations in 16-bit → uses GPU **tensor cores**, ~halves memory
bandwidth. Best for **compute-bound CNNs** (Faster R-CNN ×1.28). Transformers (DETR)
barely gain (×0.96) — they are more overhead/launch-bound than FLOP-bound at this size.
- Code: `model.half()` + cast inputs, in [torchvision_det.py](detectors/torchvision_det.py) / [hf_detr.py](detectors/hf_detr.py).

### 15.2 torch.compile (TorchInductor)
Captures the model graph and **fuses** many small ops into few kernels, cutting
Python/launch overhead. Best for models with **many tiny ops** — the DETR family's
attention blocks: **DETR ×1.82, RT-DETR ×2.16** (the biggest single win in the study).
- Code: `torch.compile(model, mode="reduce-overhead")`; first call pays a compile cost
  (absorbed by warmup). In [hf_detr.py](detectors/hf_detr.py) / [torchvision_det.py](detectors/torchvision_det.py).
- Ref: [torch.compile](https://pytorch.org/docs/stable/torch.compiler.html)

### 15.3 TensorRT (NVIDIA)
Builds an optimized **inference engine** from the graph: layer/tensor fusion, kernel
auto-tuning, fixed precision. Needs a (mostly) **static graph**, so it shines on
**YOLO** (clean CNN): ×1.32 → 76 FPS. Used two ways here:
- **YOLO**: Ultralytics native `model.export(format="engine")` → `.engine`. [yolo.py](detectors/yolo.py).
- **DETR**: export to ONNX (fixed 800×1333) → run via **onnxruntime-gpu's TensorRT
  Execution Provider** (×1.25). [hf_detr.py](detectors/hf_detr.py)`._setup_onnx`.
- RT-DETR's deformable-attention ops aren't TRT/ONNX-exportable → auto-skipped.
- Ref: [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/)

### 15.4 ONNX + onnxruntime
ONNX is a portable graph format; **onnxruntime** runs it with pluggable
**Execution Providers** (CPU / CUDA / TensorRT). Here it is the bridge that gives DETR
its GPU-accelerated rows. The model is exported at a **fixed input size** (boxes are
normalized, so post-processing with the original image size stays correct).
- Code: `torch.onnx.export` + `ort.InferenceSession(providers=[...])` in [hf_detr.py](detectors/hf_detr.py).
- Needs torch's bundled cuDNN on `LD_LIBRARY_PATH` (see README §4 / §11).
- Ref: [onnxruntime EPs](https://onnxruntime.ai/docs/execution-providers/)

### 15.5 Not wired (documented)
torch-tensorrt and OpenVINO (not installed); **vLLM** could *serve* the LocateAnything
VLM but lacks its custom architecture, so it is the serving path, not a drop-in here.

**One-line takeaway:** match the accelerator to the architecture — **CNN→FP16/TensorRT,
transformer(DETR)→torch.compile, YOLO→TensorRT**.
