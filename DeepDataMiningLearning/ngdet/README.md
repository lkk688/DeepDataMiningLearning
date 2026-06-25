# ngdet — Next-Generation Detection evaluation framework

A pluggable, **teaching-oriented** framework to evaluate any 2D detector, **zero-shot
(no training)**, across **COCO / KITTI / Waymo / nuScenes / nuImages** under one
**configurable unified label taxonomy** — measuring both **accuracy** (COCO mAP +
cross-domain generalization heatmap) and **inference latency** (native PyTorch vs
FP16 / torch.compile / TensorRT / ONNX), and rendering annotated evaluation videos.

It sits beside `detection/` and `vision/` and **reuses their code** (dataset loaders,
COCO evaluation, video drawing) rather than duplicating it.

> 👉 New here? Read **[TUTORIAL.md](TUTORIAL.md)** — it has the exact commands and the
> full benchmark results. 

## Why

- Students can compare model architectures (Faster R-CNN vs DETR vs RT-DETR vs YOLO
  vs open-vocab Grounding DINO/OWLv2 vs a VLM) on different datasets with one command,
  and **see the cross-domain generalization gap** and the **accuracy/latency trade-off**.
- It is the **zero-shot baseline harness** for the lab's research directions
  (VFM-enhanced multi-dataset detection, 2D→3D, future-box forecasting).

## Setup / Installation

### 1. Python environment

Python 3.10+ with a CUDA-enabled PyTorch. On this machine we use the `py312` conda env:

```bash
conda create -n py312 python=3.12 -y && conda activate py312
# CUDA 12.x PyTorch (match your driver; this repo was run on torch 2.10 + cu128)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 2. Core dependencies (required)

```bash
pip install transformers ultralytics pycocotools opencv-python \
            matplotlib numpy pandas pyarrow tqdm
```
- `transformers` → HF DETR / RT-DETR / Grounding DINO / OWLv2
- `ultralytics`  → YOLOv8 / YOLO11 / YOLO26 / YOLO-World
- `pycocotools`  → COCO mAP; `pyarrow` → Waymo parquet; `opencv-python` → video

### 3. Optional: VLM backend (LocateAnything)

```bash
pip install lmdb            # required by the LocateAnything custom modeling code
```
The model is pulled from `nvidia/LocateAnything-3B` with `trust_remote_code=True`.

### 4. Optional: acceleration backends (for the latency study)

```bash
# DETR ONNX / TensorRT rows via onnxruntime-gpu (use the CUDA-12 build!)
pip install "onnxruntime-gpu==1.20.1"
# YOLO TensorRT/ONNX export uses tensorrt (Ultralytics drives it):
pip install tensorrt
```
The DETR `onnx`/`tensorrt` rows need torch's bundled cuDNN on the loader path:

```bash
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$(ls -d $SP/nvidia/*/lib | tr '\n' ':')$LD_LIBRARY_PATH
```

### 5. Datasets

Default roots are in [run_eval.py](run_eval.py) `DEFAULT_ROOTS`; override with `--roots name=path`.

| dataset | what you need | notes |
|---|---|---|
| coco | `val2017/` + `annotations/instances_val2017.json` | same-domain reference |
| kitti | KITTI dir with `training/image_2` + `label_2` | native 2D boxes |
| waymo | Waymo v2 parquet dir (`training/camera_image`,`camera_box`) | native 2D; use `--waymo-stride` |
| nuscenes | a root containing a `v1.0-trainval` json folder | 2D = **projected 3D (loose)** |
| nuimages | extracted nuImages dir (`samples/` + `v1.0-*` json) | **real 2D boxes (recommended)** |

Extract nuImages (real 2D boxes):
```bash
cd <nuscenes_dir>
tar xzf nuimages-v1.0-mini.tgz       -C nuimages   # quick: 50 imgs
tar xzf nuimages-v1.0-all-metadata.tgz -C nuimages  # train/val/test json
tar xzf nuimages-v1.0-all-samples.tgz  -C nuimages  # all images (~16 GB)
```

For the nuScenes **mini** split (the full trainval JSON is huge), build a symlink root:
```python
from DeepDataMiningLearning.detection.verify_datasets_video import build_nuscenes_mini_root
build_nuscenes_mini_root("<nuscenes>/v1.0-mini", "ngdet/output/nuscenes_mini_root")
```

### 6. Verify the install

```bash
python -m DeepDataMiningLearning.ngdet.evaluator        # synthetic mAP ~1.0
python -m DeepDataMiningLearning.ngdet.detectors.base   # lists registered backends
```

## Layout

```
ngdet/
├── taxonomy.py        # configurable unified label space + name->id folding
├── detectors/
│   ├── base.py        # BaseDetector + Detection + registry (build_detector)
│   ├── torchvision_det.py  # classic Faster R-CNN / RetinaNet / FCOS
│   ├── hf_detr.py     # HuggingFace AutoModelForObjectDetection (DETR/RT-DETR/...)
│   ├── grounding_dino.py   # open-vocab Grounding DINO + OWLv2 (zero-shot)
│   ├── yolo.py        # Ultralytics YOLO11/12 (+ open-vocab YOLO-World)
│   └── locate_anything.py  # NVIDIA LocateAnything-3B (experimental, Phase 2)
├── datasets.py        # EvalDataset wrappers over detection/dataset_*.py
├── evaluator.py       # COCO mAP (pycocotools) on the unified taxonomy
├── video.py           # annotated mp4 writer (reuses detection/verify_datasets_video.py)
├── report.py          # model x dataset summary, generalization gap, heatmap PNG
├── latency.py         # native-vs-accelerated latency benchmark (fp16/compile/tensorrt/onnx)
└── run_eval.py        # the single CLI entry point
```

## Quick start (Phase 1)

Run from the **repo root** (the directory containing the `DeepDataMiningLearning` package),
with an env that has `torch transformers ultralytics pycocotools opencv-python`:

```bash
# DETR + YOLO on KITTI, 50 images, with annotated video. First run downloads weights.
python -m DeepDataMiningLearning.ngdet.run_eval \
    --models hf_detr:facebook/detr-resnet-50 yolo:yolo11x \
    --datasets kitti --max-images 50 --score-thr 0.3 --video \
    --out-dir output/ngdet
```

Add COCO as the same-domain reference:

```bash
python -m DeepDataMiningLearning.ngdet.run_eval \
    --models hf_detr:facebook/detr-resnet-50 yolo:yolo11x \
    --datasets kitti coco --max-images 200 --video \
    --coco-ann /mnt/e/Shared/Dataset/coco2017/annotations/instances_val2017.json
```

Switch the taxonomy (the label space is configurable):

```bash
--taxonomy driving3        # vehicle / person / cyclist  (default)
--taxonomy vehicle_person  # 2-class
--taxonomy driving5        # car / truck / bus / person / cyclist
```

## Per-file self-tests

Every module has a runnable `__main__` (no full pipeline needed):

```bash
python -m DeepDataMiningLearning.ngdet.taxonomy          # label folding demo
python -m DeepDataMiningLearning.ngdet.datasets --name kitti --max-images 3
python -m DeepDataMiningLearning.ngdet.evaluator         # synthetic mAP~1.0 sanity
python -m DeepDataMiningLearning.ngdet.video             # 3-frame KITTI GT clip
python -m DeepDataMiningLearning.ngdet.detectors.base    # list registered backends
```

## Add a new detector (≈50 lines)

Subclass `BaseDetector`, implement `predict()`, register it:

```python
from .base import BaseDetector, Detection, register

@register("my_backend")
class MyDetector(BaseDetector):
    def __init__(self, taxonomy, device="cuda", score_thr=0.3, model_name=None, **kw):
        super().__init__(taxonomy, device, score_thr)
        ...  # load model; build self.id_lut = taxonomy.build_id_lut(id2name)
    def predict(self, image, prompt=None) -> Detection:
        ...  # return self._fold_to_unified(boxes_xyxy, scores, native_ids)
```

Then use `--models my_backend:checkpoint`.
