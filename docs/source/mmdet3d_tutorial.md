# MMdetection3d
## Lab Server and Jetson access
Open your config file (nano ~/.ssh/config) and add the following blocks. For Windows system, the full path to the config file is: "C:\Users\<YourUserName>\.ssh\config".

```bash
Host edge-gateway
    Hostname edgeai.duckdns.org
    User vmuser

Host edgeaiserver
    Hostname 127.0.0.1
    User student
    Port 2224
    ProxyJump edge-gateway
Host jetson01
    Hostname 127.0.0.1
    User student
    Port 2222
    ProxyJump edge-gateway
```
Once you save that file, you only need to run one command to ssh into our server (same spec with HPC):
```bash
ssh edgeaiserver
```
If you want to ssh into our jetson device, only one command:
```bash
ssh jetson01
```

If you are using VScode or similar (like Google's Antigravity), you can install "Remote - SSH" extension created by Microsoft. Because you saved your config file correctly, VS Code will automatically list the hosts (`edgeaiserver` and `jetson01`). You can simply click to connect. 

The jeton tutorial can be access in this link: [sjsujetontool](https://lkk688.github.io/edgeAI/curriculum/00b_sjsujetsontool_cheatsheet/). Bascially, you can just run `sjsujetsontool shell` to enter container shell.

## mmdetection3d Installation in SJSU HPC - Step by Step Guide

Overview
- This guide installs MMDetection3D on SJSU HPC, aligned with CUDA 12.6.
- You will set up Conda, install PyTorch, MMCV (full with CUDA ops), MMEngine, and MMDet/MMDet3D.
- You’ll verify CUDA and C++/CUDA extension availability, prepare datasets, and run benchmarks.
- The HPC GPU nodes are isolated from the internet; a proxy is required when downloading models/data.

### SSH into the HPC
```bash
ssh SJSUID@coe-hpc1.sjsu.edu  # e.g., ssh 0107960xx@coe-hpc1.sjsu.edu
# password is your standard SJSU account password
```

Explanation
- SSH into the campus HPC gateway using your SJSU ID.
- This places you into a login node, where interactive work is okay but heavy compute should be done on GPU nodes.

Follow the pop up instructions to ssh into the HPC3:
```bash
ssh -X coe-hpc3
```

### Setup Python Environment
Install Miniconda (latest) via bash
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts; allow conda init
source ~/.bashrc
conda -V
```

Create a dedicated Conda env (`py310`) to avoid dependency clashes.
```bash
$ conda create -y -n py310 python=3.10
conda info --envs #check env created
conda activate py310 # activate env
```

Load CUDA module: use Preinstalled CUDA via Modules (recommended). HPC manages CUDA toolchains via environment modules. Loading `nvhpc-hpcx-cuda12/24.11` sets paths for CUDA 12.6.
```bash
$ module avail #check available modules
$ module load nvhpc-hpcx-cuda12/24.11 #load cuda module
$ nvcc --version #check cuda version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Sep_12_02:18:05_PDT_2024
Cuda compilation tools, release 12.6, V12.6.77
Build cuda_12.6.r12.6/compiler.34841621_0
```
Verify `nvcc --version` shows CUDA 12.6; this must match PyTorch CUDA (`cu126`) and any compiled MMCV extensions.

Install Pytorch that matches the CUDA version and other dependencies
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -U pip setuptools wheel
python -m pip install -U openmim
mim install mmengine
```
The `--index-url` pin ensures PyTorch uses CUDA 12.6 (`cu126`) builds, compatible with the module you loaded. MMEngine is the runtime framework required by MMDet/MMDet3D.

MMCV-full contains CUDA/C++ ops required by MMDet3D; `mmcv-lite` does not. Installing a local prebuilt wheel avoids compiling MMCV on the cluster. Install our own built version of mmcv-full under our shared folder:
```bash
pip install /data/cmpe249-fa25/mmcv-2.1.0*-linux_x86_64.whl
```
If you must compile MMCV later, see the “Installation Debug” section.

Run the following script in the command line to check the installations:
```bash
python - <<'PY'
import sys, numpy, matplotlib
print("Python     :", sys.version.split()[0])
print("NumPy      :", numpy.__version__)
print("Matplotlib :", matplotlib.__version__)
from matplotlib import pyplot as plt
print("pyplot OK")
import torch, mmengine, mmcv
print("Torch     :", torch.__version__, "| CUDA:", torch.version.cuda, "| is_available:", torch.cuda.is_available())
print("MMEngine  :", mmengine.__version__)
import pkgutil
try:
    import mmcv
    print("MMCV   :", mmcv.__version__, "at", mmcv.__file__)
    print("has mmcv._ext ? ", pkgutil.find_loader("mmcv._ext") is not None)
except Exception as e:
    print("MMCV import error:", repr(e))
PY
```
Explanation
- `mmcv._ext` being True confirms C++/CUDA ops are available; this is required by many 3D models.
- If False, you likely installed `mmcv-lite` or built MMCV incorrectly; see “Installation Debug”.

Install mmdet and mmdet3d, make sure the versions are compatible with each other.
```bash
mim install "mmdet>=3.0.0,<3.3.0" #mim install 'mmdet>=3.0.0' #
mim install "mmdet3d>=1.1.0"
pip uninstall numpy
pip install 'numpy<2'

python - <<'PY'
import torch, mmengine, mmcv, mmdet
print("Torch     :", torch.__version__, "| CUDA:", torch.version.cuda, "| is_available:", torch.cuda.is_available())
import mmdet3d
print("MMEngine  :", mmengine.__version__)
print("MMCV      :", mmcv.__version__)
print("MMDet     :", mmdet.__version__)
print("MMDet3D   :", mmdet3d.__version__)
PY
```
Pin `numpy<2` because many scientific stacks and some compiled extensions still expect NumPy < 2.0 ABI.


Download the mmdetection3d source code to your specified directory:
```bash
$ git clone https://github.com/open-mmlab/mmdetection3d.git
```

Enter into the mmdetection3d directory and download the models:
```bash
cd mmdetection3d
(py310) [mmdetection3d]$ mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest ./modelzoo_mmdetection3d/
```
These model configures are available for download:
```bash
'3dssd_4x4_kitti-3d-car', 'centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d', 'centerpoint_voxel01_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d', 'centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d', 'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d', 'centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d', 'centerpoint_pillar02_second_secfpn_head-dcn_8xb4-cyclic-20e_nus-3d', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area1.py', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area2.py', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area3.py', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area4.py', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area5.py', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area6.py', 'dv_second_secfpn_6x8_80e_kitti-3d-car', 'dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class', 'dv_pointpillars_secfpn_6x8_160e_kitti-3d-car', 'fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune', 'pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_fpn_head-free-anchor_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_regnet-400mf_fpn_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_regnet-400mf_fpn_head-free-anchor_sbn-all_8xb4-2x_nus-3d', 'hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d', 'pointpillars_hv_regnet-1.6gf_fpn_head-free-anchor_sbn-all_8xb4-strong-aug-3x_nus-3d', 'pointpillars_hv_regnet-3.2gf_fpn_head-free-anchor_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_regnet-3.2gf_fpn_head-free-anchor_sbn-all_8xb4-strong-aug-3x_nus-3d', 'groupfree3d_head-L6-O256_4xb8_scannet-seg.py', 'groupfree3d_head-L12-O256_4xb8_scannet-seg.py', 'groupfree3d_w2x-head-L12-O256_4xb8_scannet-seg.py', 'groupfree3d_w2x-head-L12-O512_4xb8_scannet-seg.py', 'h3dnet_3x8_scannet-3d-18class', 'imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class', 'imvotenet_stage2_16x8_sunrgbd-3d-10class', 'imvoxelnet_kitti-3d-car', 'monoflex_dla34_pytorch_dlaneck_gn-all_2x4_6x_kitti-mono3d', 'dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class', 'mask-rcnn_r50_fpn_1x_nuim', 'mask-rcnn_r50_fpn_coco-2x_1x_nuim', 'mask-rcnn_r50_caffe_fpn_1x_nuim', 'mask-rcnn_r50_caffe_fpn_coco-3x_1x_nuim', 'mask-rcnn_r50_caffe_fpn_coco-3x_20e_nuim', 'mask-rcnn_r101_fpn_1x_nuim', 'mask-rcnn_x101_32x4d_fpn_1x_nuim', 'cascade-mask-rcnn_r50_fpn_1x_nuim', 'cascade-mask-rcnn_r50_fpn_coco-20e_1x_nuim', 'cascade-mask-rcnn_r50_fpn_coco-20e_20e_nuim', 'cascade-mask-rcnn_r101_fpn_1x_nuim', 'cascade-mask-rcnn_x101_32x4d_fpn_1x_nuim', 'htc_r50_fpn_coco-20e_1x_nuim', 'htc_r50_fpn_coco-20e_20e_nuim', 'htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim', 'paconv_ssg_8xb8-cosine-150e_s3dis-seg.py', 'paconv_ssg-cuda_8xb8-cosine-200e_s3dis-seg', 'parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-3class', 'parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-car', 'pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d', 'pgd_r101-caffe_fpn_head-gn_16xb2-1x_nus-mono3d', 'pgd_r101-caffe_fpn_head-gn_16xb2-1x_nus-mono3d_finetune', 'pgd_r101-caffe_fpn_head-gn_16xb2-2x_nus-mono3d', 'pgd_r101-caffe_fpn_head-gn_16xb2-2x_nus-mono3d_finetune', 'point-rcnn_8xb2_kitti-3d-3class', 'pointnet2_ssg_2xb16-cosine-200e_scannet-seg-xyz-only', 'pointnet2_ssg_2xb16-cosine-200e_scannet-seg', 'pointnet2_msg_2xb16-cosine-250e_scannet-seg-xyz-only', 'pointnet2_msg_2xb16-cosine-250e_scannet-seg', 'pointnet2_ssg_2xb16-cosine-50e_s3dis-seg', 'pointnet2_msg_2xb16-cosine-80e_s3dis-seg', 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car', 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class', 'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_secfpn_sbn-all_8xb4-amp-2x_nus-3d', 'pointpillars_hv_fpn_sbn-all_8xb4-amp-2x_nus-3d', 'pointpillars_hv_secfpn_sbn-all_8xb2-2x_lyft-3d', 'pointpillars_hv_fpn_sbn-all_8xb2-2x_lyft-3d', 'pointpillars_hv_secfpn_sbn_2x16_2x_waymoD5-3d-car', 'pointpillars_hv_secfpn_sbn_2x16_2x_waymoD5-3d-3class', 'pointpillars_hv_secfpn_sbn_2x16_2x_waymo-3d-car', 'pointpillars_hv_secfpn_sbn_2x16_2x_waymo-3d-3class', 'pointpillars_hv_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d', 'pointpillars_hv_regnet-1.6gf_fpn_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_regnet-400mf_secfpn_sbn-all_2x8_2x_lyft-3d', 'pointpillars_hv_regnet-400mf_fpn_sbn-all_2x8_2x_lyft-3d', 'second_hv_secfpn_8xb6-80e_kitti-3d-car', 'second_hv_secfpn_8xb6-80e_kitti-3d-3class', 'second_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-3class', 'second_hv_secfpn_8xb6-amp-80e_kitti-3d-car', 'second_hv_secfpn_8xb6-amp-80e_kitti-3d-3class', 'smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d', 'hv_ssn_secfpn_sbn-all_16xb2-2x_nus-3d', 'hv_ssn_regnet-400mf_secfpn_sbn-all_16xb2-2x_nus-3d', 'hv_ssn_secfpn_sbn-all_16xb2-2x_lyft-3d', 'hv_ssn_regnet-400mf_secfpn_sbn-all_16xb1-2x_lyft-3d', 'votenet_8xb16_sunrgbd-3d.py', 'votenet_8xb8_scannet-3d.py', 'votenet_iouloss_8x8_scannet-3d-18class', 'minkunet18_w16_torchsparse_8xb2-amp-15e_semantickitti', 'minkunet18_w20_torchsparse_8xb2-amp-15e_semantickitti', 'minkunet18_w32_torchsparse_8xb2-amp-15e_semantickitti', 'minkunet34_w32_minkowski_8xb2-laser-polar-mix-3x_semantickitti', 'minkunet34_w32_spconv_8xb2-amp-laser-polar-mix-3x_semantickitti', 'minkunet34_w32_spconv_8xb2-laser-polar-mix-3x_semantickitti', 'minkunet34_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti', 'minkunet34_w32_torchsparse_8xb2-laser-polar-mix-3x_semantickitti', 'minkunet34v2_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti', 'cylinder3d_4xb4-3x_semantickitti', 'cylinder3d_8xb2-laser-polar-mix-3x_semantickitti', 'pv_rcnn_8xb2-80e_kitti-3d-3class', 'fcaf3d_2xb8_scannet-3d-18class', 'fcaf3d_2xb8_sunrgbd-3d-10class', 'fcaf3d_2xb8_s3dis-3d-5class', 'spvcnn_w16_8xb2-amp-15e_semantickitti', 'spvcnn_w20_8xb2-amp-15e_semantickitti', 'spvcnn_w32_8xb2-amp-15e_semantickitti', 'spvcnn_w32_8xb2-amp-laser-polar-mix-3x_semantickitti'
```

`mim download` fetches the exact config and checkpoint needed for testing. Store them under a local folder (e.g., `modelzoo_mmdetection3d/`) for consistency. Run the demo script to verify the installation:
```bash
(py310) [mmdetection3d]$ python demo/pcd_demo.py demo/data/kitti/000008.bin modelzoo_mmdetection3d/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py modelzoo_mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth --out-dir demo/results
```

Explanation
- This local demo quickly checks runtime correctness using KITTI sample data and a known checkpoint.
- The output should include visualizations saved under `demo/results`.

### Request one GPU node
Submit an interactive GPU job through Slurm (`gpuqs` partition). Do all heavy work (data processing, evaluation, training) on your assigned GPU node.
```bash
srun -p gpuqs --pty /bin/bash
```


After you got the GPU node, check the GPU status:
```bash
nvidia-smi
```
Confirm CUDA devices are visible and drivers load successfully.


The internal GPU node does not have internet access, set internet proxy environment variables to enable the internet access for the GPU node (you can run the command line or put in `~/.bashrc`):
```bash
export http_proxy=http://172.16.1.2:3128
export https_proxy=http://172.16.1.2:3128
curl --proxy http://172.16.1.2:3128 "https://www.sjsu.edu"  # git works
curl "https://www.sjsu.edu"  # also works
```

Explanation
- GPU nodes are behind a firewall without direct internet. Configure proxies to allow `git`, `pip`, and `curl` to work.
- You can add these to `~/.bashrc` if you need persistent access.

### Run benchmark evaluation

Run the benchmark evaluation inside the mmdetection3d directory and make sure the python environment is activated.

Create a symlink to the actual dataset location (e.g., data/kitti or data/nuscenes inside the mmdetection3d folder). 
```bash
# 1) Set your repo path
MMDETECTION3D=/yourpath/mmdetection3d

# 2) Create data dir
mkdir -p "$MMDETECTION3D/data"

# 3) Symlinks (idempotent: -s=symlink, -n=treat link as file, -f=overwrite link)
ln -snf /data/cmpe249-fa25/nuscenes/v1.0-trainval      "$MMDETECTION3D/data/nuScenes"
ln -snf /data/cmpe249-fa25/kitti         "$MMDETECTION3D/data/kitti"
ln -snf /data/cmpe249-fa25/waymov143_individuals         "$MMDETECTION3D/data/waymo"

# 4) Verify
ls -l "$MMDETECTION3D/data"
```

This benchmark evaluation runs on the validation split of the NuScenes dataset:
```bash
(py310) mmdetection3d]$ python tools/test.py ./modelzoo_mmdetection3d/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py ./modelzoo_mmdetection3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth
```

Explanation
- `tools/test.py` runs evaluation on the validation split using the config and checkpoint you downloaded.
- Ensure datasets (NuScenes, KITTI) are correctly linked under the expected `data/` directory.

## mmdetection3d Quick Installation
Simple process
```bash
#new setup in python3.10
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


## Dataset Preparation
```bash
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
(py310) lkk688@newalienware:~/Developer/mmdetection3d$ ln -snf /mnt/e/Shared/Dataset/NuScenes/v1.0-trainval ./data/nuScenes
(py310) lkk688@newalienware:~/Developer/mmdetection3d$ python tools/create_data.py nuscenes --root-path ./data/nuScenes --out-dir ./data/nuScenes --extra-tag nuscenes
```

Explanation
- Create a symlink to the actual dataset location. The directory name must match expected case in the config (NuScenes vs nuscenes).
- `tools/create_data.py` pre-processes metadata and builds info files used by dataloaders.
- This step can take a long time; make sure your node allocation is sufficient.

```bash
mmdetection3d]$ python demo/pcd_demo.py demo/data/kitti/000008.bin pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth --out-dir

(py310) [010796032@g21 mmdetection3d]$ PYTHONPATH=. python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
Generate info. this may take several minutes.
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3712/3712, 42.7 task/s, elapsed: 87s, ETA:     0s
Kitti info train file is saved to data/kitti/kitti_infos_train.pkl
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3769/3769, 44.6 task/s, elapsed: 84s, ETA:     0s
Kitti info val file is saved to data/kitti/kitti_infos_val.pkl
Kitti info trainval file is saved to data/kitti/kitti_infos_trainval.pkl
Kitti info test file is saved to data/kitti/kitti_infos_test.pkl
create reduced point cloud for training set
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3712/3712, 130.6 task/s, elapsed: 28s, ETA:     0s
create reduced point cloud for validation set
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3769/3769, 119.4 task/s, elapsed: 32s, ETA:     0s
create reduced point cloud for testing set
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 7518/7518, 32.4 task/s, elapsed: 232s, ETA:     0s
./data/kitti/kitti_infos_train.pkl will be modified.
Warning, you may overwriting the original data ./data/kitti/kitti_infos_train.pkl.
Reading from input file: ./data/kitti/kitti_infos_train.pkl.
Start updating:
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3712/3712, 190.1 task/s, elapsed: 20s, ETA:     0s
Writing to output file: ./data/kitti/kitti_infos_train.pkl.
ignore classes: {np.str_('DontCare')}
./data/kitti/kitti_infos_val.pkl will be modified.
Warning, you may overwriting the original data ./data/kitti/kitti_infos_val.pkl.
Reading from input file: ./data/kitti/kitti_infos_val.pkl.
Start updating:
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3769/3769, 186.0 task/s, elapsed: 20s, ETA:     0s
Writing to output file: ./data/kitti/kitti_infos_val.pkl.
ignore classes: {np.str_('DontCare')}
./data/kitti/kitti_infos_trainval.pkl will be modified.
Warning, you may overwriting the original data ./data/kitti/kitti_infos_trainval.pkl.
Reading from input file: ./data/kitti/kitti_infos_trainval.pkl.
Start updating:
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 7481/7481, 188.7 task/s, elapsed: 40s, ETA:     0s
Writing to output file: ./data/kitti/kitti_infos_trainval.pkl.
ignore classes: {np.str_('DontCare')}
./data/kitti/kitti_infos_test.pkl will be modified.
Warning, you may overwriting the original data ./data/kitti/kitti_infos_test.pkl.
Reading from input file: ./data/kitti/kitti_infos_test.pkl.
Start updating:
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 7518/7518, 4663.0 task/s, elapsed: 2s, ETA:     0s
Writing to output file: ./data/kitti/kitti_infos_test.pkl.
ignore classes: set()
Create GT Database of KittiDataset
11/07 13:59:05 - mmengine - INFO - ------------------------------
11/07 13:59:05 - mmengine - INFO - The length of training dataset: 3712
11/07 13:59:05 - mmengine - INFO - The number of instances per category in the dataset:
+----------------+--------+
| category       | number |
+----------------+--------+
| Pedestrian     | 2207   |
| Cyclist        | 734    |
| Car            | 14357  |
| Van            | 1297   |
| Truck          | 488    |
| Person_sitting | 56     |
| Tram           | 224    |
| Misc           | 337    |
+----------------+--------+
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3712/3712, 57.4 task/s, elapsed: 65s, ETA:     0s
load 2207 Pedestrian database infos
load 14357 Car database infos
load 734 Cyclist database infos
load 1297 Van database infos
load 488 Truck database infos
load 224 Tram database infos
load 337 Misc database infos
load 56 Person_sitting database infos
```

Explanation
- KITTI preprocessing creates info PKLs and a GT database for augmentation.
- These logs confirm successful metadata generation and class distributions.

## mmdetection3d Installation Debug
If the above process does not work, run these:
```bash
$ conda create -y -n py310 python=3.10
conda activate py310
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -U pip setuptools wheel
python -m pip install -U openmim
mim install mmengine
mim install 'mmcv<2.2.0,>=2.0.0rc4'

python - <<'PY'
import sys, numpy, matplotlib
print("Python     :", sys.version.split()[0])
print("NumPy      :", numpy.__version__)
print("Matplotlib :", matplotlib.__version__)
from matplotlib import pyplot as plt
print("pyplot OK")
import torch, mmengine, mmcv
print("Torch     :", torch.__version__, "| CUDA:", torch.version.cuda, "| is_available:", torch.cuda.is_available())
print("MMEngine  :", mmengine.__version__)
import pkgutil
try:
    import mmcv
    print("MMCV   :", mmcv.__version__, "at", mmcv.__file__)
    print("has mmcv._ext ? ", pkgutil.find_loader("mmcv._ext") is not None)
except Exception as e:
    print("MMCV import error:", repr(e))
PY
```

Explanation
- If MMCV C++/CUDA ops fail, you may be on a PyTorch/CUDA combo without prebuilt MMCV wheels.
- The fix is to compile MMCV against your current CUDA toolkit (matching `nvcc` version).
- Ensure `CC`/`CXX` point to GCC/G++, not NVHPC compilers.

```bash
# Ensure you have full CUDA toolkit 12.6 installed first!
# nvcc -V  <-- must verify this works and matches 12.6

git clone https://github.com/open-mmlab/mmcv.git
cd mmcv

#switch to v2.1.0
git fetch --tags
git checkout v2.1.0

# 1. Cleanup previous attempts
rm -rf build/ mmcv.egg-info/

# 2. Force standard GCC compilers
export CC=gcc
export CXX=g++

# 3. Set CUDA paths specifically for version 12.6
export CUDA_HOME=/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6
export CPATH=$CPATH:$CUDA_HOME/targets/x86_64-linux/include
export LIBRARY_PATH=$LIBRARY_PATH:$CUDA_HOME/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/targets/x86_64-linux/lib
export PATH=$CUDA_HOME/bin:$PATH

# 4. Verify setup before building
echo "CUDA_HOME is $CUDA_HOME"
nvcc -V

# 5. Retry compilation
MMCV_WITH_OPS=1 python setup.py build_ext --inplace

#6.  Install it
pip install .

#7. (optional) save your MMCV 2.2.0 build
python setup.py bdist_wheel
#in the dist/ folder
mmcv]$ cp ./dist/mmcv-2.1.0-cp310-cp310-linux_x86_64.whl /data/cmpe249-fa25

pip install dist/mmcv-2.1.0*-linux_x86_64.whl --force-reinstall
```

Explanation
- `MMCV_WITH_OPS=1` ensures CUDA ops are compiled.
- If compilation errors mention missing headers like `cusparse.h`, point `CPATH`/`LIBRARY_PATH` to CUDA Math libraries (see BEVFusion notes later).

```bash
mim install 'mmdet>=3.0.0' #pip install "mmdet>=3.0.0,<3.3.0"
mim install "mmdet3d>=1.1.0"
pip uninstall numpy
pip install 'numpy<2'

python - <<'PY'
import torch, mmengine, mmcv, mmdet
print("Torch     :", torch.__version__, "| CUDA:", torch.version.cuda, "| is_available:", torch.cuda.is_available())
import mmdet3d
print("MMEngine  :", mmengine.__version__)
print("MMCV      :", mmcv.__version__)
print("MMDet     :", mmdet.__version__)
print("MMDet3D   :", mmdet3d.__version__)
PY


python - <<'PY'
import sys, pkgutil, torch
print("Python :", sys.version.split()[0])
print("Torch  :", torch.__version__, "CUDA:", torch.version.cuda, "avail:", torch.cuda.is_available())
try:
    import mmcv
    print("MMCV   :", mmcv.__version__, "at", mmcv.__file__)
    print("has mmcv._ext ? ", pkgutil.find_loader("mmcv._ext") is not None)
except Exception as e:
    print("MMCV import error:", repr(e))
PY
```

show the following results:
```bash
Python : 3.10.19
Torch  : 2.9.0+cu126 CUDA: 12.6 avail: True
MMCV   : 2.1.0 at /fs/atipa/data/rnd-liu/MyRepo/mmcv/mmcv/__init__.py
has mmcv._ext ?  True
```

Explanation
- Validates that MMCV is now correctly installed with CUDA ops on this environment.

```bash
$ ln -s /data/rnd-liu/Datasets/nuScenes/v1.0-trainval ./data/nuscenes
(py310) [010796032@g21 mmdetection3d]$ PYTHONPATH=. python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
======
Loading NuScenes tables for version v1.0-trainval...
23 category,
8 attribute,
4 visibility,
64386 instance,
12 sensor,
10200 calibrated_sensor,
2631083 ego_pose,
68 log,
850 scene,
34149 sample,
2631083 sample_data,
1166187 sample_annotation,
4 map,
Done loading in 34.669 seconds.
======
Reverse indexing ...
Done reverse indexing in 8.0 seconds.
======
total scene num: 850
exist scene num: 850
train scene: 700, val scene: 150
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 34149/34149, 18.0 task/s, elapsed: 1898s, ETA:     0s
train sample: 28130, val sample: 6019
./data/nuscenes/nuscenes_infos_train.pkl will be modified.
Warning, you may overwriting the original data ./data/nuscenes/nuscenes_infos_train.pkl.
Reading from input file: ./data/nuscenes/nuscenes_infos_train.pkl.
======
Loading NuScenes tables for version v1.0-trainval...
23 category,
8 attribute,
4 visibility,
64386 instance,
12 sensor,
10200 calibrated_sensor,
2631083 ego_pose,
68 log,
850 scene,
34149 sample,
2631083 sample_data,
1166187 sample_annotation,
4 map,
Done loading in 39.650 seconds.
======
Reverse indexing ...
Done reverse indexing in 8.3 seconds.
======
Start updating:
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 28130/28130, 12.3 task/s, elapsed: 2282s, ETA:     0s
Writing to output file: ./data/nuscenes/nuscenes_infos_train.pkl.
ignore classes: set()
./data/nuscenes/nuscenes_infos_val.pkl will be modified.
Warning, you may overwriting the original data ./data/nuscenes/nuscenes_infos_val.pkl.
Reading from input file: ./data/nuscenes/nuscenes_infos_val.pkl.
======
Loading NuScenes tables for version v1.0-trainval...
23 category,
8 attribute,
4 visibility,
64386 instance,
12 sensor,
10200 calibrated_sensor,
2631083 ego_pose,
68 log,
850 scene,
34149 sample,
2631083 sample_data,
1166187 sample_annotation,
4 map,
Done loading in 43.734 seconds.
======
Reverse indexing ...
Done reverse indexing in 8.2 seconds.
======
Start updating:
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 6019/6019, 13.7 task/s, elapsed: 438s, ETA:     0s
Writing to output file: ./data/nuscenes/nuscenes_infos_val.pkl.
ignore classes: set()
Create GT Database of NuScenesDataset
11/07 15:42:38 - mmengine - INFO - ------------------------------
11/07 15:42:38 - mmengine - INFO - The length of training dataset: 28130
11/07 15:42:38 - mmengine - INFO - The number of instances per category in the dataset:
+----------------------+--------+
| category             | number |
+----------------------+--------+
| car                  | 413318 |
| truck                | 72815  |
| trailer              | 20701  |
| bus                  | 13163  |
| construction_vehicle | 11993  |
| bicycle              | 9478   |
| motorcycle           | 10109  |
| pedestrian           | 185847 |
| traffic_cone         | 82362  |
| barrier              | 125095 |
```

Explanation
- NuScenes preprocessing is heavy and may take hours depending on I/O and CPU.
- Ensure paths are correct and you have read permissions to the dataset location.

```bash
$ pip3 install waymo-open-dataset-tf-2-12-0==1.6.7
pip install -U typing_extensions
(py310) [010796032@g21 mmdetection3d]$ PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=3 python tools/create_data.py waymo --root-path ./data/waymo --out-dir ./data/waymo --workers 128 --extra-tag waymo --version v1.4
```

Explanation
- Waymo tools rely on TensorFlow 2.12 wheels and specific protobuf versions; the `TF_CPP_MIN_LOG_LEVEL=3` reduces TF logging noise.
- Adjust `--workers` based on CPU availability on the node.

```bash
(py310) [010796032@g21 mmdetection3d]$ mim download mmdet3d --config pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d --dest ./modelzoo_mmdete
ction3d/
processing pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d...
downloading ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.9/18.9 MiB 14.3 MB/s eta 0:00:00
Successfully downloaded hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth to /fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/modelzoo_mmdetection3d
Successfully dumped pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py to /fs/atipa/data/rnd-liu/MyRepo/mmdetection3d/modelzoo_mmdetection3d

(py310) [010796032@g21 mmdetection3d]$ python tools/test.py ./modelzoo_mmdetection3d/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py ./modelzoo_mmdetection3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth

Results writes to /tmp/tmpqlq6hvx7/results/pred_instances_3d/results_nusc.json
Evaluating bboxes of pred_instances_3d
mAP: 0.3369                                                                                                                                
mATE: 0.4264
mASE: 0.2847
mAOE: 0.5304
mAVE: 0.3876
mAAE: 0.2001
NDS: 0.4855
Eval time: 76.7s
```

NuScenes Metrics Explained
- mAP: Mean Average Precision across classes and distance thresholds.
- mATE: Translation error (meters); lower is better.
- mASE: Scale error; lower is better.
- mAOE: Orientation error; lower is better.
- mAVE: Velocity error; lower is better.
- mAAE: Attribute error (e.g., moving/static).
- NDS: NuScenes Detection Score (combines metrics, higher is better).
- Per-class AP/ATE/etc. indicate performance per object type (car, truck, pedestrian, etc.).

```bash
python - <<'PY'
import pickle as pkl
p="/data/rnd-liu/MyRepo/mmdetection3d/data/nuscenes/nuscenes_infos_train.pkl"
with open(p,"rb") as f:
    obj=pkl.load(f)
print(type(obj), list(obj)[:2])  # v1:list[...]  v2:dict with 'data_list'/metadata
print("PKL load OK")
PY
```

Explanation
- Verifies that NuScenes info files are readable and their structure matches expectations (list vs dict depending on version).

BEVFusion, https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion:
```bash
(py310) [010796032@g21 mmdetection3d]$ export CUDA_HOME=/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6
export CPATH=$CPATH:$CUDA_HOME/targets/x86_64-linux/include
export LIBRARY_PATH=$LIBRARY_PATH:$CUDA_HOME/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/targets/x86_64-linux/lib
export PATH=$CUDA_HOME/bin:$PATH
(py310) [010796032@g21 mmdetection3d]$ python projects/BEVFusion/setup.py develop

S__ --expt-relaxed-constexpr --compiler-options '-fPIC' -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=voxel_layer -std=c++17
In file included from /home/010796032/miniconda3/envs/py310/lib/python3.10/site-packages/torch/include/ATen/cuda/CUDAContext.h:3,
                 from projects/BEVFusion/bevfusion/ops/voxel/src/scatter_points_cuda.cu:2:
/home/010796032/miniconda3/envs/py310/lib/python3.10/site-packages/torch/include/ATen/cuda/CUDAContextLight.h:8:10: fatal error: cusparse.h: No such file or directory
    8 | #include <cusparse.h>
      |          ^~~~~~~~~~~~
compilation terminated.
error: command '/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6/bin/nvcc' failed with exit code 1

(py310) [010796032@g21 mmdetection3d]$ find /opt/ohpc -name cusparse.h 2>/dev/null

# 2. Define where the math libraries are hiding
export MATH_LIB_HOME=/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/math_libs/12.6/targets/x86_64-linux

# 3. Add them to your standard search paths
# (CPATH for .h headers, LIBRARY_PATH for .so linking)
export CPATH=$CPATH:$MATH_LIB_HOME/include
export LIBRARY_PATH=$LIBRARY_PATH:$MATH_LIB_HOME/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MATH_LIB_HOME/lib
```

Explanation
- The error indicates CUDA Math libraries (like `cusparse.h`) are not in your include/library paths.
- Adding `MATH_LIB_HOME/include` and `.../lib` to CPATH/LIBRARY_PATH/LD_LIBRARY_PATH fixes headers and linker resolution.

```bash
Creating /fs/atipa/home/010796032/miniconda3/envs/py310/lib/python3.10/site-packages/bev-pool.egg-link (link to .)
Adding bev-pool 0.0.0 to easy-install.pth file

(py310) [010796032@g21 mmdetection3d]$ python projects/BEVFusion/setup.py develop
Installed /fs/atipa/data/rnd-liu/MyRepo/mmdetection3d
Processing dependencies for bev-pool==0.0.0
Finished processing dependencies for bev-pool==0.0.0

(py310) [010796032@g21 mmdetection3d]$ python tools/test.py projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py ./modelzoo_mmdetection3d/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth

(py310) [010796032@g21 mmdetection3d]$ python tools/test_insecure.py projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py ./modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth

#The checkpoint was likely trained with an older or different version of mmcv/spconv where convolution weights had a shape like (out_C, k, k, k, in_C). Your current model expects (k, k, k, in_C, out_C).
#bev_pool CUDA kernel was compiled for a different GPU architecture than the one you are currently running on.

export TORCH_CUDA_ARCH_LIST="8.6 8.9 9.0"
cd projects/BEVFusion
rm -rf build/
pip install -U ninja cmake

# 2) Make sure you’re using GNU compilers (not NVHPC’s nvc/nvc++)
export CC=$(command -v gcc)
export CXX=$(command -v g++)
which gcc; which g++; which nvcc

(py310) [010796032@g21 mmdetection3d]$ python projects/BEVFusion/setup.py build_ext --inplace
python projects/BEVFusion/setup.py develop
```

Explanation
- If checkpoints were trained under different `spconv` layouts, kernel shapes may mismatch. The fix below permutes OC-first → KD×KH×KW×IC×OC.
- Set `TORCH_CUDA_ARCH_LIST` to match your GPU architectures; recompile BEVFusion CUDA ops with GNU compilers.

```bash
(py310) [010796032@g21 mmdetection3d]$ python tools/test_insecure.py projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py ./modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth

#that mismatch pattern screams spconv v1 ↔ v2 kernel layout drift.
#current mmdet3d + spconv v2 expects KD×KH×KW×IC×OC, while the BEVFusion checkpoint stores OC×KD×KH×KW×IC. 
python - <<'PY'
import torch, os

# === path in/out ===
src = "./modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth"
dst = "./modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.fixed_spconvv2.pth"

ckpt = torch.load(src, map_location="cpu", weights_only=False)
sd = ckpt.get("state_dict", ckpt)  # some ckpts store weights at top-level

def needs_permute(t):
    # spconv-v1-style weights look like [OC, KD, KH, KW, IC].
    # If the first dim isn't a typical kernel size (1/3/5/7), assume OC-first and permute.
    return isinstance(t, torch.Tensor) and t.ndim == 5 and t.shape[0] not in (1,3,5,7)

n_perm = 0
for k, v in list(sd.items()):
    if needs_permute(v):
        sd[k] = v.permute(1, 2, 3, 4, 0).contiguous()  # [OC,KD,KH,KW,IC] -> [KD,KH,KW,IC,OC]
        n_perm += 1

if "state_dict" in ckpt:
    ckpt["state_dict"] = sd
else:
    ckpt = sd

torch.save(ckpt, dst)
print(f"Permuted {n_perm} 3D conv kernels. Saved to: {dst}")
PY

export PYTHONPATH=$PWD:$PYTHONPATH
python tools/test_insecure.py \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  ./modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.fixed_spconvv2.pth

Formating bboxes of pred_instances_3d
Start to convert detection format...
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 6019/6019, 17.1 task/s, elapsed: 352s, ETA:     0s
Results writes to /tmp/tmpmk9bf1y4/results/pred_instances_3d/results_nusc.json
Evaluating bboxes of pred_instances_3d
mAP: 0.6841                                                                                                                                
mATE: 0.2796
mASE: 0.2546
mAOE: 0.3005
mAVE: 0.2806
mAAE: 0.1879
NDS: 0.7117
Eval time: 79.5s

Per-class results:
Object Class            AP      ATE     ASE     AOE     AVE     AAE   
```

Explanation
- After fixing kernel layouts and recompiling, metrics significantly improve; verify NDS, mAP are in expected ranges.
- Use simple sanity checks (e.g., visualize predictions) to confirm model outputs look reasonable.

Common Pitfalls & Tips
- If `mmcv._ext` is False, confirm you’re not using `mmcv-lite`. Reinstall MMCV-full or compile from source.
- Ensure CUDA toolkit and PyTorch CUDA versions match (`nvcc --version` aligns with `torch.version.cuda`).
- Pin `numpy<2` to avoid ABI issues with compiled ops in some stacks.
- On GPU nodes, configure proxy variables for `pip`, `git`, and `curl`.
- For BEVFusion, set `TORCH_CUDA_ARCH_LIST` and use GCC/G++ for building CUDA extensions; clean `build/` before recompilation.

# New Design

Analysis: Why Your NDS is Below Original BEVFusion
Root causes in your current configs
1. Fully frozen Swin backbone (frozen_stages=4)
The original BEVFusion fine-tunes Swin-Small at all stages. You freeze Swin-Tiny entirely. This is the single biggest NDS gap. No domain adaptation = the image features never converge to nuScenes statistics.

2. Short training schedule
v9: 6 epochs, v12: 10 epochs. Original: 20 epochs. CrossAttnLSS needs more iterations to learn good depth distributions.

3. GridMask disabled (prob=0.0)
Original BEVFusion uses GridMask as a key regularizer. Yours is on but at 0 probability — no effect.

4. Camera BEV bottleneck
out_channels=128/160 vs 256 in original. The camera branch has less representational capacity going into ConvFuser.

5. Partial LiDAR backbone freeze
FreezeExceptHook only unlocks encoder_layers.0 and .1 of the SECOND backbone. Layers 2+ are frozen, limiting LiDAR feature adaptation.

6. EMA commented out
EMA (momentum=0.0002) is a low-cost +0.5–1.0 NDS regularizer used in most top models.

Immediate Fixes (Expected +2–4 NDS)
Fix	Config Change	Expected Gain
Unfreeze Swin last 2 stages	frozen_stages=2, add 'img_backbone.layers.2': dict(lr_mult=0.1) etc.	+1–2 NDS
Train 20 epochs	max_epochs=20, extend scheduler	+0.5–1.5
Enable GridMask	prob=0.7	+0.3–0.8
Increase camera BEV to 256ch	out_channels=256 in view_transform, in_channels=[256,256] in fusion_layer	+0.5–1
Re-enable EMA	uncomment EMAHook	+0.3–0.5
Unfreeze full SECOND backbone	remove layers 0/1 restriction from allowlist	+0.3
Add depth supervision	depth GT loss on CrossAttnLSS (like BEVDepth)	+0.5–1


Research Contributions (Novel for a Paper)
Your actual novel contributions are solid: CrossAttnLSS view transform + VoxelPainting with gating + FireRPF necks at lower GPU cost. To make the paper publishable, frame these plus the following improvements:

A. Depth-supervised CrossAttnLSS
Add a depth distribution supervision head on the CrossAttnLSS view transformer (like BEVDepth does for LSS). Your cross-attention query structure makes this natural since attention weights implicitly encode depth — you can supervise them directly with LiDAR-projected depth. This is your unique angle over vanilla BEVDepth.

B. Bidirectional Cross-Modal Fusion
Replace ConvFuser with a lightweight bidirectional attention block: camera BEV attends to LiDAR BEV and vice versa before concatenation. SparseFusion and CMT show this closes the gap vs simple concat fusion. This pairs well with your existing FireRPF attention (CBAM) story.

C. Temporal BEV Aggregation (Strong NDS boost)
The original BEVFusion is single-frame. Adding a simple recurrent BEV memory (even a single-step ego-aligned feature warp) typically gives +3–5 NDS on nuScenes. BEVFusion4D and Lift-Splat-Temporal variants show this. Since you already have a custom BEV pipeline, temporal fusion is a clean extension that fits the "efficient" narrative.

D. Knowledge-Distilled Camera Branch
Use your trained LiDAR-BEV features as a teacher to distill the camera BEV branch (like UniDistill or BEVDistill). This works well when combined with your gated voxel painting — the painting provides the bridge between modalities for distillation.


New Research Directions in Autonomous Driving (for Paper Positioning)
1. 3D Occupancy Prediction (Hottest current area)
Replace or augment 3D object detection with dense occupancy prediction (OpenOccupancy, Occ3D, SurroundOcc). Your memory-efficient BEV pipeline is well-positioned here since occupancy heads add volume without needing bigger backbones. This extends your contribution from "better detection" to "richer scene representation."

2. End-to-End Sensor Fusion for Planning
Papers like UniAD, SparseDrive, and VAD connect detection → tracking → motion prediction → planning in one model. Your efficient fusion backbone could serve as the perception module for an E2E planner — this is a high-impact framing for a system paper.

3. Radar + LiDAR + Camera Fusion
Most nuScenes-based work ignores radar. Adding radar as a third modality (nuScenes has it) gives a unique contribution angle. Your gated voxel painting architecture generalizes to painting radar returns onto LiDAR voxels. CRN and RCBEVDet have shown +1–3 NDS from radar fusion with minor memory overhead.

4. Efficient 4D BEV with Sparse Representations
Work like SparseBEV, FocalFormer3D, and HEDNet move away from dense BEV grids to sparse queries. Since your paper's strength is memory efficiency, adopting sparse BEV tokenization at the fusion stage would be a coherent architectural story — merge your CBAM-gated RPF necks with sparse deformable attention.

5. Foundation Model Backbones
Replacing Swin with DINOv2 or InternImage (pre-trained at scale) and freezing only early layers gives substantially better camera features. Several 2024–2025 papers (e.g., Far3D, MonoDETR variants) show +3–5 NDS purely from this backbone swap while maintaining low fine-tuning cost.


Suggested Paper Narrative
"Memory-Efficient Multi-Modal BEVFusion with Cross-Attention View Transform and Gated Feature Painting"

Claim: Achieve competitive NDS to BEVFusion (within 1–2 points) at X% lower peak GPU memory through (1) CrossAttnLSS replacing depth-LSS, (2) lightweight FireRPF necks replacing SECFPN, and (3) gated voxel painting replacing dense image features. Add temporal fusion or depth supervision as the "+1 novel contribution."

The memory efficiency story is publishable — AAAI/IROS/ITSC all value practical deployment metrics. The gap to close is proving the accuracy isn't sacrificed: the fixes in the table above should get you within 1–2 NDS of the original.

————————
Important Clarification: MindDrive
MindDrive is a Vision-Language-Action (VLA) model — it uses an LLM with LoRA adapters trained on the Bench2Drive dataset for linguistic driving decisions. It has no BEV encoder and is not suitable for direct integration with your pipeline. For trajectory/planning that plugs into BEV features, use the alternatives below.

Complete Perception Pipeline Architecture
Your CrossAttnLSS BEVFusion backbone produces a shared fused BEV feature map [B, 256, 180, 180]. All three tasks consume this same tensor — one encoder, three heads.


┌─────────────────────────────────────────────────────────────────────┐
│                    INPUT SENSORS                                     │
│    6x Cameras [B,6,3,256,704]    LiDAR Points [B, N, 5]            │
└───────────────┬──────────────────────────┬──────────────────────────┘
                │                          │
     ┌──────────▼──────────┐    ┌──────────▼──────────┐
     │  Swin-Tiny + FPN    │    │  SECOND + FireRPF   │
     │  + FireRPF2DNeck    │    │  LiDAR BEV          │
     │  + CrossAttnLSS     │    │  [B, 256, 180, 180] │
     │  Camera BEV         │    └──────────┬──────────┘
     │  [B, 160, 180, 180] │               │
     └──────────┬──────────┘               │
                │                          │
     ┌──────────▼──────────────────────────▼──────────┐
     │          ConvFuser  (your existing code)        │
     │          Fused BEV [B, 256, 180, 180]          │
     │          ← SHARED FEATURE BACKBONE →           │
     └──────┬──────────────┬─────────────────┬────────┘
            │              │                 │
   ┌────────▼────┐  ┌──────▼──────┐  ┌──────▼──────────┐
   │  Detection  │  │  Occupancy  │  │   HD Map Head   │
   │  Head       │  │  Head       │  │  (MapTRv2 dec.) │
   │  (existing) │  │  (new)      │  │  (new)          │
   └────────┬────┘  └──────┬──────┘  └──────┬──────────┘
            │              │                 │
   10-class  3D boxes  3D Occ grid     Vectorized lanes
   NDS metric  [B,W,H,Z]  semantic      + dividers
                           voxels        + crossings
            │              │                 │
            └──────────────┴─────────────────┘
                           │
              ┌────────────▼────────────────┐
              │   Lightweight BEV Planner   │
              │   (VAD or SparseDrive)      │
              │   Trajectory [B, T, 2]      │
              └─────────────────────────────┘

-------

Task 1: 3D Occupancy Head (Add to existing model)
Recommended: TPVFormer-style occupancy or FB-OCC

The simplest path: lift your 2D BEV features [B, 256, 180, 180] to 3D voxels with a lightweight Z-deconvolution head.


# Conceptual occupancy head (add to BEVFusionCA)
occ_head = dict(
    type='BEVOccHead',
    in_channels=256,
    num_classes=17,            # nuScenes-occ 16 classes + empty
    bev_h=180, bev_w=180,
    num_z_anchors=16,          # 8m z range / 0.5m = 16 voxels
    z_range=[-5.0, 3.0],
    # Lift BEV → 3D via learned z-deconv, ~2M params
    z_deconv_channels=[256, 128, 64],
    loss_occ=dict(type='CrossEntropyLoss', use_sigmoid=False,
                  class_weight=[...]),   # reweight rare classes
    loss_lovasz=dict(type='LovaszLoss', reduction='none'),
)
The Lovász loss is critical for occupancy — cross-entropy alone under-weights thin structures like cyclists and cones.

Dataset: nuScenes-Occupancy (Occ3D-nuScenes, 700 train/150 val scenes, same splits as your detection training). You get both tasks from one training run.

Task 2: HD Map Head
Recommended: MapTRv2 (simpler than InteractionMap, well-maintained codebase)


# MapTRv2 decoder plugs directly onto fused BEV
map_head = dict(
    type='MapTRv2Head',
    bev_h=180, bev_w=180,
    bev_dim=256,               # matches your fused BEV
    num_vec=50,                # max map element instances
    num_pts_per_vec=20,        # polyline points per element
    num_classes=3,             # divider, crossing, boundary
    pc_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
    # Transformer decoder (6 layers, lightweight)
    transformer=dict(
        type='MapTRPerceptionTransformer',
        decoder=dict(num_layers=6, ...),
    ),
    loss_cls=dict(type='FocalLoss', ...),
    loss_pts=dict(type='PtsL1Loss', ...),
    loss_dir=dict(type='PtsDirCosLoss', ...),
)
Training: MapTRv2 on nuScenes needs ~110 epochs standalone, but when your BEV backbone is pre-trained from detection, 30–40 additional epochs suffice since the BEV features are already good.

Task 3: Trajectory Planning
Recommended: VAD (ICLR 2024) or BEV-Planner — NOT MindDrive

VAD uses vectorized scene representations (your map outputs + detected objects) to plan. It's the right fit because:

Takes BEV features + map vectors + detected boxes — all outputs you already have
~30M parameters, runs at 15+ FPS on GPU
nuScenes planning L2/collision metrics
Minimal integration: add a GRU-based ego-motion planner on top of your fused BEV:


planner_head = dict(
    type='SimpleBEVPlanner',
    bev_channels=256,
    ego_channels=64,             # ego speed/heading embedding
    future_steps=6,              # 3s at 2Hz
    num_modes=3,                 # multimodal trajectories
    loss_traj=dict(type='L1Loss'),
    loss_collision=dict(type='BCELoss', weight=10.0),
)
For the paper, you can use the nuScenes Open-loop Planning benchmark (L2 displacement + collision rate) as the third evaluation metric alongside NDS and mAP-map.

Multi-Task Training Strategy

# In your BEVFusionCA model, add multi-task loss weighting
# Use uncertainty weighting (Kendall et al. 2018) or fixed weights:

loss_weights = dict(
    det_loss=1.0,          # your existing detection loss
    occ_loss=1.0,          # new: occupancy CE + Lovász
    map_loss=1.0,          # new: MapTRv2 cls + pts + dir
    plan_loss=0.5,         # new: L2 + collision
)
Training schedule on H100:

Phase 1 (10 epochs): Detection + Occupancy only (share BEV encoder)
Phase 2 (10 epochs): Add Map head (freeze occ head)
Phase 3 (10 epochs): Full multi-task with planner (all unfrozen, lower LR)
Total: ~30 epochs for the full pipeline vs. 20 epochs for detection alone.

Jetson Thor Optimization Plan
Jetson AGX Thor has Tensor Cores with INT8/FP16 support and is designed for AV workloads. Here's the deployment pipeline:

Step 1: Model Surgery Before Export

# Replace non-TRT-friendly ops:
# CrossAttnLSS: replace torch.einsum with explicit matmul
# CBAM: replace adaptive_avg_pool2d with fixed-size pool
# spconv: use torch-tensorrt or replace with dense conv for export
Step 2: TensorRT Export via torch2trt or TensorRT Python API

# FP16 for backbone + necks (no accuracy loss)
# INT8 for detection/occ heads (calibrate on 512 nuScenes samples)
# Keep MapTRv2 decoder in FP16 (attention ops sensitive to INT8)
Step 3: Module-level Latency Targets (estimated)
Module	H100 FP16	Jetson Thor FP16	Jetson Thor INT8
Swin-Tiny + FPN	~8ms	~35ms	~20ms
CrossAttnLSS VT	~12ms	~50ms	~30ms
SECOND + FireRPF	~5ms	~20ms	~12ms
ConvFuser	~1ms	~3ms	~2ms
Det head	~2ms	~8ms	~5ms
Occ head	~3ms	~12ms	~7ms
MapTRv2 decoder	~5ms	~20ms	~14ms
Total	~36ms	~148ms	~90ms
Target: ~10 FPS on Thor with INT8 — achievable. Real AV systems run perception at 10Hz.

Step 4: Thor-Specific Optimizations
Sparse convolution: spconv 2.x with TensorRT plugin support for the SECOND backbone — avoids materializing empty voxels
Chunked attention: your existing attn_chunk setting in CrossAttnLSS is already the right pattern for memory-constrained inference
BEV resolution reduction at inference: drop to [128, 128] BEV grid at test time on Thor with minimal NDS loss (~0.5 NDS), halving memory and compute
Occupancy at lower Z resolution: 8 Z levels instead of 16 for Thor inference
Suggested Paper Title and Contribution
"UniDriveBEV: A Memory-Efficient Unified BEV Perception Pipeline for 3D Detection, Occupancy, Mapping, and Planning"

Contributions:

CrossAttnLSS view transform as a drop-in LSS replacement (memory efficient, GPU-agnostic)
Single shared BEV encoder for 4 tasks simultaneously — no redundant feature extraction
FireRPF necks as lightweight CBAM-gated alternatives to SECFPN
Jetson Thor INT8 deployment analysis with throughput/accuracy trade-off study
This positions the paper at IROS 2026 or ITSC 2026 (system/deployment papers), or ECCV 2026 Workshop on Autonomous Driving if the NDS gap is closed.



--------

What was built: bevdet/train/
All files live at DeepDataMiningLearning/bevdet/train/ and are accessible as projects.bevdet.train.* from the mmdetection3d directory.

File map
File	Purpose	mmdet3d dependency
misc.py	Seeds, AverageMeter, ETA, Lovász softmax	None
checkpoint.py	EMA, save/load full state, weights-only load	None
occ_head.py	BEVOccHead + LiDAR pseudo-occ builder	None
multitask_bev.py	MultiTaskBEVFusion (det + occ hook)	Registry only
trainer.py	Trainer class (BF16, AdamW, cosine LR)	None
eval_runner.py	Correct NDS via Runner.test()	Runner only
train.py	Main entry point	Config + Registry
radar_stub.py	RadarPillarEncoder stub for Phase 2	None
configs/mybevfusion_phase1.py	Phase 1 config (extends v12v2)	Config inherit
Launch commands

conda activate py310
cd /data/rnd-liu/MyRepo/mmdetection3d

# Phase 1a: detection only (verify parity with original tools/train.py)
python projects/bevdet/train/train.py \
    --config projects/bevdet/configs/mybevfusion12v2.py \
    --work-dir work_dirs/phase1_det \
    --epochs 10 \
    --eval-interval 2

# Phase 1b: detection + binary occupancy (no Occ3D dataset required)
python projects/bevdet/train/train.py \
    --config projects/bevdet/configs/mybevfusion_phase1.py \
    --work-dir work_dirs/phase1_det_occ \
    --epochs 10 \
    --occ-classes 2 \
    --occ-num-z 16 \
    --occ-loss-weight 0.5 \
    --eval-interval 2

# Evaluate only (always correct NDS)
python projects/bevdet/train/train.py \
    --config projects/bevdet/configs/mybevfusion12v2.py \
    --eval-only \
    --load-from work_dirs/phase1_det/epoch_10.pth \
    --data-root /data/rnd-liu/MyRepo/mmdetection3d/data/nuscenes \
    --work-dir work_dirs/phase1_det
Key design decisions
Eval always uses Runner.test() — the manual NuScenes eval path is removed from the training loop. This fixes the NDS gap you observed.
Per-epoch checkpoints save det-only weights (epoch_N.pth) — so the Runner loads them without the occ_head wrapper. Full multitask weights saved separately as epoch_N_multitask.pth.
Occupancy GT is optional — if occ_gt is absent from a batch, occ loss is silently skipped. This lets you start training detection-only and add occ labels incrementally.
Radar — RadarPillarEncoder is a stub with a clear interface contract. Flip use_radar=True in Phase 2 without touching detection or occupancy code.


```bash
cd /fs/atipa/data/rnd-liu/MyRepo/mmdetection3d && conda run -n py310 python projects/bevdet/train/train.py \
    --config projects/bevdet/configs/mybevfusion12v2.py \
    --load-from work_dirs/mybevfusion12v2/epoch_8.pth \
    --work-dir work_dirs/phase1a_correcttest \
    --epochs 2 \
    --warmup-iters 200 \
    --log-interval 200 \
    --debug-iters 400 \
    --eval-interval 0 \
    --workers 8 \
    --seed 42 2>&1 | grep -v "UserWarning\|warnings.warn\|^$\|mmengine - INFO\|mmengine - DEBUG" | tail -35
```

Full Phase 1a training is now running:

Config: mybevfusion12v2.py
Pretrained weights: work_dirs/mybevfusion12v2/epoch_8.pth
Output: work_dirs/phase1a_full/
Schedule: 10 epochs, 7724 iters/epoch ≈ ~28h total at 3.3s/iter

-------

Phase 1b: What changed and why

Change	File	Expected NDS gain
frozen_stages 4→2 (Swin stages 2&3 fine-tuned at lr 0.02x)	mybevfusion_phase1b.py	+1.5–2.5 pts
Camera BEV channels 160→256	same	+0.3–0.8 pts
GridMask prob 0.0→0.7	same	+0.3–0.7 pts
EMAHook enabled	same	+0.1–0.3 pts
20 epochs (was 10)	same + run_phase1b.sh	+0.5–1.0 pts
Binary occupancy auxiliary loss	run_phase1b.sh --occ-classes 2	+0.2–0.5 pts
Target: NDS ~0.70–0.72 vs current 0.68

Bug fixes made:

checkpoint.py: load_weights now skips shape-mismatched tensors (prints warning) instead of crashing — needed whenever architecture changes between checkpoints
multitask_bev.py: MultiTaskBEVFusion.__init__ now accepts both a BEVOccHead instance and a dict for occ_head
Once the smoke test passes (background task), run the full 20-epoch training with:


cd /data/rnd-liu/MyRepo/mmdetection3d
bash run_phase1b.sh

Training is running (main PID 913453 at 125% CPU, 8 data workers). Here's a summary of all fixes applied before the successful launch:

3 bugs fixed to get Phase 1b running:

Shape mismatch crash — load_weights in checkpoint.py now filters out tensors whose shapes changed (view_transform 160→256, fusion_layer 416→512 input), prints a warning, and loads the rest. Required any time architecture changes between checkpoints.

occ_head type error — multitask_bev.py now accepts both a BEVOccHead instance and a dict; train.py passes an already-constructed instance.

CUDA/CPU device mismatch on mean — Det3DDataPreprocessor.device property uses next(self.buffers()) which can return a CPU-side voxelizer buffer before reaching mean, causing cast_data to leave images on CPU while mean is on CUDA. Fixed in train.py build_model by explicitly calling model.data_preprocessor.to(device) after model.to(device).

Phase 1b is now running: 20 epochs, backbone unfrozen (stages 2-3 at lr 0.02x), GridMask enabled, 256-ch camera BEV, EMA, + binary occupancy aux loss. Target NDS: ~0.70-0.72 vs current 0.68.

Monitor with:


tail -f work_dirs/phase1b/train.log | grep -E "Epoch(train)|mean_loss|Checkpoint|Error"