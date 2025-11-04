

```bash
$ conda create --name py310 python=3.10 -y
$ module avail
$ module load nvhpc-hpcx-cuda12/24.11
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Sep_12_02:18:05_PDT_2024
Cuda compilation tools, release 12.6, V12.6.77
Build cuda_12.6.r12.6/compiler.34841621_0
$ conda activate py310
(py310) $ pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

```

Install Waymo dataset package
```bash
(py310) $ pip3 install waymo-open-dataset-tf-2-12-0==1.6.7
#test imports
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import v2
```

Install mmdetection3d
```bash
#new setup
pip install -U openmim
mim install mmengine
pip uninstall numpy
pip install 'numpy<2'
mim install 'mmcv<2.2.0,>=2.0.0rc4'
Successfully installed mmcv-2.1.0
mim install 'mmdet>=3.0.0'
$ git clone https://github.com/open-mmlab/mmdetection3d.git
(py310) lkk688@newalienware:~/Developer/mmdetection3d$ pip install -v -e .
>>> import mmdet3d
>>> print(mmdet3d.__version__)
1.4.0
>>> import mmdet
>>> print(mmdet.__version__)
3.3.0

```