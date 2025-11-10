# HPC

## Author
- Kaikai Liu, Associate Professor, SJSU  
- Email: kaikai.liu@sjsu.edu  
- Web: http://www.sjsu.edu/cmpe/faculty/tenure-line/kaikai-liu.php

## Introduction of HPC
You can access our CoE HPC information from the main website: http://coe-hpc-web.sjsu.edu.

- HPC1: A total of 36 nodes; 15 include 1 NVIDIA Tesla P100 12 GB GPU, and 1 has 2 NVIDIA Tesla P100 GPUs
- New A100 and H100 nodes are added in HPC2 and HPC3
- Data storage, available from all nodes via `/home` and `/data`
- High-throughput Lustre parallel filesystem with 0.541 PB usable space via `/scratch`
- Each group has a subdirectory in `/data` and `/scratch`

If you provided your SJSU ID to your instructor, you can access the HPC using your SJSU account:

- SSH: `ssh SJSUID@coe-hpc1.sjsu.edu` (replace `SJSUID` with your ID)
- Accessible group folders: `/data/cmpe249-fa25` and `/scratch/cmpe249-fa25` (example class ID)
- Create subdirectories in one of these group directories for your project
- Place datasets in `/data/cmpe249-fa25`.
- The head node is not for heavy computation or large storage; keep home dir under 20 GB
- GPU/CPU nodes are available upon request; courtesy resource with no guarantee
- Some frameworks cache in your home dir; set different cache folders for PyTorch and Hugging Face in `~/.bashrc` (below)

---

## Remote Access to HPC

### SSH Access
To SSH into the HPC host machine, first establish a campus VPN connection: https://www.sjsu.edu/it/services/network/vpn/index.php.
Your SJSU ID should already have HPC access via your instructor/advisor. For access issues, request via your advisor; do not contact campus IT.

```bash
ssh SJSUID@coe-hpc1.sjsu.edu  # e.g., ssh 0107960xx@coe-hpc1.sjsu.edu
# password is your standard SJSU account password
```

Follow the pop up instructions to ssh into the HPC3:
```bash
ssh -X coe-hpc3
```

### File Transfer
You have several options for transferring files to and from HPC:

- Use `scp` for secure copy
- JupyterLab’s file browser for uploads/downloads
- Use `sshfs` to mount HPC data folders on your local machine (Linux):

```bash
sudo apt-get install sshfs
sshfs <sjsuid>@coe-hpc1.sjsu.edu:/data/cmpeXXX .
```

### VSCode Remote to HPC
Use Visual Studio Code Remote (https://code.visualstudio.com/docs/remote/ssh):

- Install the “Remote - SSH” extension in VS Code
- Add a new SSH connection and open `coe-hpc1`
- Open your desired working directory
- You can open Jupyter notebooks and use Git to sync with GitHub (head node only)
- VS Code has been tested on the GPU node and is functional in HPC2 (non-functional in HPC1). Enable dual-hop SSH. Git sync is not available on GPU nodes due to no internet.
- Add the following to your `~/.ssh/config` (open via VSCode Remote Explorer):

```bash
Host coe-hpc1
  HostName coe-hpc1.sjsu.edu
  ForwardAgent yes
  User 0107xxxxxx  # your SJSU ID
  ForwardX11 yes
  ForwardX11Trusted yes
  ServerAliveInterval 30
```

### Bashrc Setup (optional)
Set proxies and cache folders:

```bash
nano ~/.bashrc
# add the following lines
export http_proxy=http://172.16.1.2:3128
export https_proxy=http://172.16.1.2:3128
export HF_HOME=/data/cmpe249-fa25/[XXXcache]/huggingface
export TORCH_HOME=/data/cmpe249-fa25/[XXXcache]/torch
source ~/.bashrc  # to take effect
```
---

## VSCode Remote to Lab’s Server
Dual-hop SSH to the HPC1 GPU node isn’t operational. Alternatively, use dual-hop SSH to the lab server “hpc1p100” in ENG276 (NVIDIA P100 GPU, internet access, conventional Linux server). Use HPC1 GPU node for long-running jobs.

Add to `~/.ssh/config`:

```bash
Host hpc1p100
  Hostname 130.65.157.216
  User student
  ForwardX11 yes
  ForwardX11Trusted yes
  ServerAliveInterval 30
  ProxyCommand ssh coe-hpc1 -W %h:%p
```

Access the lab server:

```bash
ssh hpc1p100
# You will be prompted for your HPC headnode password first,
# then the lab server's 'student' account password (request from Prof. Liu).
```

On the lab server:

- Existing data or your own data can be accessed/placed under `/DATA5T2/Datasets/`
- You can mount the HPC folder to `/data/cmpe249-fa23` (write permission added via `sudo chmod 777`)
- Mount example using `sshfs`:

```bash
# on lab server
sshfs <sjsuid>@coe-hpc1.sjsu.edu:/data/cmpe249-fa23 /data/cmpe249-fa23
ls /data/cmpe249-fa23
```

---

## GPU Node Internet Access
Set proxy environment variables (can put in `~/.bashrc`):

```bash
export http_proxy=http://172.16.1.2:3128
export https_proxy=http://172.16.1.2:3128
curl --proxy http://172.16.1.2:3128 "https://www.sjsu.edu"  # git works
curl "https://www.sjsu.edu"  # also works
```

---

## X11 Window Forwarding
X11 forwarding allows running GUIs from HPC on your local machine. Tested and working with Matplotlib and OpenCV (terminal and VSCode).

- macOS: install `xQuartz` from https://www.xquartz.org/
- Linux: typically no prerequisites
- Windows: use MobaXterm (https://mobaxterm.mobatek.net/download-home-edition.html);
  or PuTTY with Xming (https://sourceforge.net/projects/xming/)
- After local setup, SSH to HPC server with X11:

```bash
ssh -Y 010xxxxxxx@coe-hpc1.sjsu.edu
```

---

## Load Software Modules and Request GPU Node
Check available modules and load required ones on the head node:

```bash
module avail
```

Conda environments:

```bash
conda info --envs        # list available conda environments
conda activate mycondapy311
```

### Request a Node with Slurm
Request a CPU node (interactive bash):

```bash
srun --pty /bin/bash
# ...
exit  # exit the computing node when done
```

Request a GPU node (interactive bash):

```bash
PARTITION   Max TIMELIMIT   Comments
========    ==============  =======================================
defq             4:00:00    4 hours - Default CPU queue
cpuqs        05-01:00:00    5 days  -   Short CPU queue
cpuqm        14-01:00:00   14 days  -  Medium CPU queue
cpuql        21-01:00:00   21 days  -    Long CPU queue
gpuqs        02-01:00:00    2 days  -   Short GPU queue
gpuqm        07-01:00:00    7 days  -  Medium GPU queue
gpuql        14-01:00:00   14 days  -    Long GPU queue
condo        30-01:00:00   30 days  -   condo GPU - Faculty queue
chemq        30-01:00:00   30 days  -   cchem GPU - Faculty queue

========================>>>>>>> NOTICE <<<<<<<<====================

        NSF Campus CyberInfrastructure Grant

PARTITION   Max TIMELIMIT   Comments: GPU nodes
========    ==============  =======================================
nsfqs        up 2-01:00:00      2 days     Short GPU queue
nsfqm        up 14-01:00:0     14 days    Medium GPU queue
nsfql        up 21-01:00:0     21 days      Long GPU queue

srun -p gpuqs --pty /bin/bash
nvidia-smi

exit #exit the GPU node if you are not using
```

Note: if you see `srun: job <id> queued and waiting for resources`, wait until `allocated resources` message appears. You will be automatically logged into the allocated GPU.

### Optional: Load TensorRT Library

```bash
conda activate mycondapy311
export LD_LIBRARY_PATH=/data/cmpe249-fa25/mycuda/TensorRT-8.4.2.4/lib:$LD_LIBRARY_PATH
```

---

## JupyterLab Access
GPU nodes do not have internet access. To access Jupyter in your local browser, set up SSH tunnels from your local machine → HPC headnode, then from headnode → GPU node. Change port `10001` to an available port.

```bash
# local → headnode
ssh -L 10001:localhost:10001 0107xxx@coe-hpc1.sjsu.edu
# headnode → GPU node
ssh -L 10001:localhost:10001 0107xxx@g7
# activate your Python environment
jupyter lab --no-browser --port=10001
```

Open the URL shown in the terminal in your local browser.

Note: if Jupyter picks a different port, the previous port might be occupied.

---

## Conda Environment Setup Tutorial
You can install Miniconda via bash or use `module load anaconda/3.9`.

### Install Miniconda (latest) via bash

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts; allow conda init
source ~/.bashrc
conda -V
conda info --envs
```

### Create a New Conda Environment

```bash
conda create --name mycondapy311  # optionally specify python=3.11
conda activate mycondapy311
conda deactivate
```

### Install JupyterLab in Conda (headnode only)

```bash
conda activate mycondapy311
conda install -c conda-forge jupyterlab
conda install ipykernel
jupyter kernelspec list
ipython kernel install --user --name=mycondapy311
```

---

## CUDA Setup Tutorial
Multiple options to install CUDA on HPC:

### Option 1: Use Preinstalled CUDA via Modules (recommended)

```bash
$ module avail
$ module load nvhpc-hpcx-cuda12/24.11
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Sep_12_02:18:05_PDT_2024
Cuda compilation tools, release 12.6, V12.6.77
Build cuda_12.6.r12.6/compiler.34841621_0
```

### Option 2: Install CUDA under Conda

```bash
conda activate mycondapy311
conda install -c conda-forge cudatoolkit=11.8.0
# Optional: CUDA development kit for nvcc
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
nvcc -V
```

---

## PyTorch Installation
Install PyTorch with matched CUDA version:

```bash
(py310) $ pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m torch.utils.collect_env
```


## Additional Projects and Tools

### Hugging Face

```bash
pip install transformers
pip install datasets
pip install sentencepiece
pip install scikit-learn
pip install accelerate
pip install evaluate
pip install xformers  # may adjust torch and CUDA-related packages
pip install umap-learn
```
### Waymo Open Dataset
Install Waymo dataset package in Python 3.10
```bash
(py310) $ pip3 install waymo-open-dataset-tf-2-12-0==1.6.7
#test imports
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import v2
```

### mmdetection3d
```bash
(py310) lkk688@newalienware:~/Developer/mmdetection3d$ ln -snf /mnt/e/Shared/Dataset/NuScenes/v1.0-trainval ./data/nuScenes
(py310) lkk688@newalienware:~/Developer/mmdetection3d$ python tools/create_data.py nuscenes --root-path ./data/nuScenes --out-dir ./data/nuScenes --extra-tag nuscenes
```

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
Verify if the MMCV C++/CUDA extensions (mmcv._ext) are working. If 'mmcv._ext' is False, means the C++/cuda libraries are not compiled and you have installed mmcv-lite (formerly mmcv) instead of mmcv (full version with CUDA ops). The most likely reason is PyTorch version (2.9.0+cu126), which is a very recent build. OpenMMLab does not yet provide pre-built binaries (wheels) for PyTorch 2.9. We need to compile MMCV. This requires a full CUDA toolkit (nvcc) installed on your system matching your PyTorch CUDA version (12.6).
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


```bash
$ pip3 install waymo-open-dataset-tf-2-12-0==1.6.7
pip install -U typing_extensions
(py310) [010796032@g21 mmdetection3d]$ PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=3 python tools/create_data.py waymo --root-path ./data/waymo --out-dir ./data/waymo --workers 128 --extra-tag waymo --version v1.4
```

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

Per-class results:
Object Class            AP      ATE     ASE     AOE     AVE     AAE   
car                     0.787   0.208   0.157   0.150   0.268   0.206 
truck                   0.384   0.431   0.209   0.208   0.259   0.231 
bus                     0.519   0.453   0.196   0.255   0.655   0.249 
trailer                 0.245   0.688   0.225   0.462   0.304   0.209 
construction_vehicle    0.059   0.845   0.439   1.454   0.145   0.345 
pedestrian              0.567   0.175   0.280   0.431   0.312   0.083 
motorcycle              0.202   0.316   0.311   0.906   0.751   0.221 
bicycle                 0.008   0.332   0.326   0.850   0.405   0.058 
traffic_cone            0.179   0.270   0.395   nan     nan     nan   
barrier                 0.419   0.547   0.310   0.055   nan     nan   
11/07 20:24:36 - mmengine - INFO - Epoch(test) [6019/6019]    NuScenes metric/pred_instances_3d_NuScenes/car_AP_dist_0.5: 0.6729  NuScenes metric/pred_instances_3d_NuScenes/car_AP_dist_1.0: 0.7930  NuScenes metric/pred_instances_3d_NuScenes/car_AP_dist_2.0: 0.8328  NuScenes metric/pred_instances_3d_NuScenes/car_AP_dist_4.0: 0.8510  NuScenes metric/pred_instances_3d_NuScenes/car_trans_err: 0.2076  NuScenes metric/pred_instances_3d_NuScenes/car_scale_err: 0.1573  NuScenes metric/pred_instances_3d_NuScenes/car_orient_err: 0.1505  NuScenes metric/pred_instances_3d_NuScenes/car_vel_err: 0.2682  NuScenes metric/pred_instances_3d_NuScenes/car_attr_err: 0.2061  NuScenes metric/pred_instances_3d_NuScenes/mATE: 0.4264  NuScenes metric/pred_instances_3d_NuScenes/mASE: 0.2847  NuScenes metric/pred_instances_3d_NuScenes/mAOE: 0.5304  NuScenes metric/pred_instances_3d_NuScenes/mAVE: 0.3876  NuScenes metric/pred_instances_3d_NuScenes/mAAE: 0.2001  NuScenes metric/pred_instances_3d_NuScenes/truck_AP_dist_0.5: 0.1873  NuScenes metric/pred_instances_3d_NuScenes/truck_AP_dist_1.0: 0.3727  NuScenes metric/pred_instances_3d_NuScenes/truck_AP_dist_2.0: 0.4647  NuScenes metric/pred_instances_3d_NuScenes/truck_AP_dist_4.0: 0.5098  NuScenes metric/pred_instances_3d_NuScenes/truck_trans_err: 0.4315  NuScenes metric/pred_instances_3d_NuScenes/truck_scale_err: 0.2088  NuScenes metric/pred_instances_3d_NuScenes/truck_orient_err: 0.2084  NuScenes metric/pred_instances_3d_NuScenes/truck_vel_err: 0.2591  NuScenes metric/pred_instances_3d_NuScenes/truck_attr_err: 0.2311  NuScenes metric/pred_instances_3d_NuScenes/trailer_AP_dist_0.5: 0.0161  NuScenes metric/pred_instances_3d_NuScenes/trailer_AP_dist_1.0: 0.1688  NuScenes metric/pred_instances_3d_NuScenes/trailer_AP_dist_2.0: 0.3384  NuScenes metric/pred_instances_3d_NuScenes/trailer_AP_dist_4.0: 0.4573  NuScenes metric/pred_instances_3d_NuScenes/trailer_trans_err: 0.6878  NuScenes metric/pred_instances_3d_NuScenes/trailer_scale_err: 0.2253  NuScenes metric/pred_instances_3d_NuScenes/trailer_orient_err: 0.4621  NuScenes metric/pred_instances_3d_NuScenes/trailer_vel_err: 0.3044  NuScenes metric/pred_instances_3d_NuScenes/trailer_attr_err: 0.2085  NuScenes metric/pred_instances_3d_NuScenes/bus_AP_dist_0.5: 0.2220  NuScenes metric/pred_instances_3d_NuScenes/bus_AP_dist_1.0: 0.5135  NuScenes metric/pred_instances_3d_NuScenes/bus_AP_dist_2.0: 0.6566  NuScenes metric/pred_instances_3d_NuScenes/bus_AP_dist_4.0: 0.6837  NuScenes metric/pred_instances_3d_NuScenes/bus_trans_err: 0.4526  NuScenes metric/pred_instances_3d_NuScenes/bus_scale_err: 0.1962  NuScenes metric/pred_instances_3d_NuScenes/bus_orient_err: 0.2553  NuScenes metric/pred_instances_3d_NuScenes/bus_vel_err: 0.6554  NuScenes metric/pred_instances_3d_NuScenes/bus_attr_err: 0.2485  NuScenes metric/pred_instances_3d_NuScenes/construction_vehicle_AP_dist_0.5: 0.0000  NuScenes metric/pred_instances_3d_NuScenes/construction_vehicle_AP_dist_1.0: 0.0136  NuScenes metric/pred_instances_3d_NuScenes/construction_vehicle_AP_dist_2.0: 0.0856  NuScenes metric/pred_instances_3d_NuScenes/construction_vehicle_AP_dist_4.0: 0.1350  NuScenes metric/pred_instances_3d_NuScenes/construction_vehicle_trans_err: 0.8453  NuScenes metric/pred_instances_3d_NuScenes/construction_vehicle_scale_err: 0.4385  NuScenes metric/pred_instances_3d_NuScenes/construction_vehicle_orient_err: 1.4544  NuScenes metric/pred_instances_3d_NuScenes/construction_vehicle_vel_err: 0.1450  NuScenes metric/pred_instances_3d_NuScenes/construction_vehicle_attr_err: 0.3449  NuScenes metric/pred_instances_3d_NuScenes/bicycle_AP_dist_0.5: 0.0045  NuScenes metric/pred_instances_3d_NuScenes/bicycle_AP_dist_1.0: 0.0079  NuScenes metric/pred_instances_3d_NuScenes/bicycle_AP_dist_2.0: 0.0081  NuScenes metric/pred_instances_3d_NuScenes/bicycle_AP_dist_4.0: 0.0106  NuScenes metric/pred_instances_3d_NuScenes/bicycle_trans_err: 0.3317  NuScenes metric/pred_instances_3d_NuScenes/bicycle_scale_err: 0.3260  NuScenes metric/pred_instances_3d_NuScenes/bicycle_orient_err: 0.8500  NuScenes metric/pred_instances_3d_NuScenes/bicycle_vel_err: 0.4054  NuScenes metric/pred_instances_3d_NuScenes/bicycle_attr_err: 0.0581  NuScenes metric/pred_instances_3d_NuScenes/motorcycle_AP_dist_0.5: 0.1453  NuScenes metric/pred_instances_3d_NuScenes/motorcycle_AP_dist_1.0: 0.2118  NuScenes metric/pred_instances_3d_NuScenes/motorcycle_AP_dist_2.0: 0.2208  NuScenes metric/pred_instances_3d_NuScenes/motorcycle_AP_dist_4.0: 0.2307  NuScenes metric/pred_instances_3d_NuScenes/motorcycle_trans_err: 0.3163  NuScenes metric/pred_instances_3d_NuScenes/motorcycle_scale_err: 0.3110  NuScenes metric/pred_instances_3d_NuScenes/motorcycle_orient_err: 0.9060  NuScenes metric/pred_instances_3d_NuScenes/motorcycle_vel_err: 0.7513  NuScenes metric/pred_instances_3d_NuScenes/motorcycle_attr_err: 0.2211  NuScenes metric/pred_instances_3d_NuScenes/pedestrian_AP_dist_0.5: 0.5405  NuScenes metric/pred_instances_3d_NuScenes/pedestrian_AP_dist_1.0: 0.5545  NuScenes metric/pred_instances_3d_NuScenes/pedestrian_AP_dist_2.0: 0.5743  NuScenes metric/pred_instances_3d_NuScenes/pedestrian_AP_dist_4.0: 0.5990  NuScenes metric/pred_instances_3d_NuScenes/pedestrian_trans_err: 0.1748  NuScenes metric/pred_instances_3d_NuScenes/pedestrian_scale_err: 0.2795  NuScenes metric/pred_instances_3d_NuScenes/pedestrian_orient_err: 0.4313  NuScenes metric/pred_instances_3d_NuScenes/pedestrian_vel_err: 0.3122  NuScenes metric/pred_instances_3d_NuScenes/pedestrian_attr_err: 0.0828  NuScenes metric/pred_instances_3d_NuScenes/traffic_cone_AP_dist_0.5: 0.1409  NuScenes metric/pred_instances_3d_NuScenes/traffic_cone_AP_dist_1.0: 0.1553  NuScenes metric/pred_instances_3d_NuScenes/traffic_cone_AP_dist_2.0: 0.1830  NuScenes metric/pred_instances_3d_NuScenes/traffic_cone_AP_dist_4.0: 0.2380  NuScenes metric/pred_instances_3d_NuScenes/traffic_cone_trans_err: 0.2696  NuScenes metric/pred_instances_3d_NuScenes/traffic_cone_scale_err: 0.3948  NuScenes metric/pred_instances_3d_NuScenes/traffic_cone_orient_err: nan  NuScenes metric/pred_instances_3d_NuScenes/traffic_cone_vel_err: nan  NuScenes metric/pred_instances_3d_NuScenes/traffic_cone_attr_err: nan  NuScenes metric/pred_instances_3d_NuScenes/barrier_AP_dist_0.5: 0.1452  NuScenes metric/pred_instances_3d_NuScenes/barrier_AP_dist_1.0: 0.4098  NuScenes metric/pred_instances_3d_NuScenes/barrier_AP_dist_2.0: 0.5371  NuScenes metric/pred_instances_3d_NuScenes/barrier_AP_dist_4.0: 0.5831  NuScenes metric/pred_instances_3d_NuScenes/barrier_trans_err: 0.5466  NuScenes metric/pred_instances_3d_NuScenes/barrier_scale_err: 0.3100  NuScenes metric/pred_instances_3d_NuScenes/barrier_orient_err: 0.0554  NuScenes metric/pred_instances_3d_NuScenes/barrier_vel_err: nan  NuScenes metric/pred_instances_3d_NuScenes/barrier_attr_err: nan  NuScenes metric/pred_instances_3d_NuScenes/NDS: 0.4855  NuScenes metric/pred_instances_3d_NuScenes/mAP: 0.3369  data_time: 0.0627  time: 0.2135
```


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

```bash
dict_keys(['3dssd_4x4_kitti-3d-car', 'centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d', 'centerpoint_voxel01_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d', 'centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d', 'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d', 'centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d', 'centerpoint_pillar02_second_secfpn_head-dcn_8xb4-cyclic-20e_nus-3d', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area1.py', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area2.py', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area3.py', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area4.py', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area5.py', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area6.py', 'dv_second_secfpn_6x8_80e_kitti-3d-car', 'dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class', 'dv_pointpillars_secfpn_6x8_160e_kitti-3d-car', 'fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune', 'pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_fpn_head-free-anchor_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_regnet-400mf_fpn_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_regnet-400mf_fpn_head-free-anchor_sbn-all_8xb4-2x_nus-3d', 'hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d', 'pointpillars_hv_regnet-1.6gf_fpn_head-free-anchor_sbn-all_8xb4-strong-aug-3x_nus-3d', 'pointpillars_hv_regnet-3.2gf_fpn_head-free-anchor_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_regnet-3.2gf_fpn_head-free-anchor_sbn-all_8xb4-strong-aug-3x_nus-3d', 'groupfree3d_head-L6-O256_4xb8_scannet-seg.py', 'groupfree3d_head-L12-O256_4xb8_scannet-seg.py', 'groupfree3d_w2x-head-L12-O256_4xb8_scannet-seg.py', 'groupfree3d_w2x-head-L12-O512_4xb8_scannet-seg.py', 'h3dnet_3x8_scannet-3d-18class', 'imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class', 'imvotenet_stage2_16x8_sunrgbd-3d-10class', 'imvoxelnet_kitti-3d-car', 'monoflex_dla34_pytorch_dlaneck_gn-all_2x4_6x_kitti-mono3d', 'dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class', 'mask-rcnn_r50_fpn_1x_nuim', 'mask-rcnn_r50_fpn_coco-2x_1x_nuim', 'mask-rcnn_r50_caffe_fpn_1x_nuim', 'mask-rcnn_r50_caffe_fpn_coco-3x_1x_nuim', 'mask-rcnn_r50_caffe_fpn_coco-3x_20e_nuim', 'mask-rcnn_r101_fpn_1x_nuim', 'mask-rcnn_x101_32x4d_fpn_1x_nuim', 'cascade-mask-rcnn_r50_fpn_1x_nuim', 'cascade-mask-rcnn_r50_fpn_coco-20e_1x_nuim', 'cascade-mask-rcnn_r50_fpn_coco-20e_20e_nuim', 'cascade-mask-rcnn_r101_fpn_1x_nuim', 'cascade-mask-rcnn_x101_32x4d_fpn_1x_nuim', 'htc_r50_fpn_coco-20e_1x_nuim', 'htc_r50_fpn_coco-20e_20e_nuim', 'htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim', 'paconv_ssg_8xb8-cosine-150e_s3dis-seg.py', 'paconv_ssg-cuda_8xb8-cosine-200e_s3dis-seg', 'parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-3class', 'parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-car', 'pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d', 'pgd_r101-caffe_fpn_head-gn_16xb2-1x_nus-mono3d', 'pgd_r101-caffe_fpn_head-gn_16xb2-1x_nus-mono3d_finetune', 'pgd_r101-caffe_fpn_head-gn_16xb2-2x_nus-mono3d', 'pgd_r101-caffe_fpn_head-gn_16xb2-2x_nus-mono3d_finetune', 'point-rcnn_8xb2_kitti-3d-3class', 'pointnet2_ssg_2xb16-cosine-200e_scannet-seg-xyz-only', 'pointnet2_ssg_2xb16-cosine-200e_scannet-seg', 'pointnet2_msg_2xb16-cosine-250e_scannet-seg-xyz-only', 'pointnet2_msg_2xb16-cosine-250e_scannet-seg', 'pointnet2_ssg_2xb16-cosine-50e_s3dis-seg', 'pointnet2_msg_2xb16-cosine-80e_s3dis-seg', 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car', 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class', 'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_secfpn_sbn-all_8xb4-amp-2x_nus-3d', 'pointpillars_hv_fpn_sbn-all_8xb4-amp-2x_nus-3d', 'pointpillars_hv_secfpn_sbn-all_8xb2-2x_lyft-3d', 'pointpillars_hv_fpn_sbn-all_8xb2-2x_lyft-3d', 'pointpillars_hv_secfpn_sbn_2x16_2x_waymoD5-3d-car', 'pointpillars_hv_secfpn_sbn_2x16_2x_waymoD5-3d-3class', 'pointpillars_hv_secfpn_sbn_2x16_2x_waymo-3d-car', 'pointpillars_hv_secfpn_sbn_2x16_2x_waymo-3d-3class', 'pointpillars_hv_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d', 'pointpillars_hv_regnet-1.6gf_fpn_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_regnet-400mf_secfpn_sbn-all_2x8_2x_lyft-3d', 'pointpillars_hv_regnet-400mf_fpn_sbn-all_2x8_2x_lyft-3d', 'second_hv_secfpn_8xb6-80e_kitti-3d-car', 'second_hv_secfpn_8xb6-80e_kitti-3d-3class', 'second_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-3class', 'second_hv_secfpn_8xb6-amp-80e_kitti-3d-car', 'second_hv_secfpn_8xb6-amp-80e_kitti-3d-3class', 'smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d', 'hv_ssn_secfpn_sbn-all_16xb2-2x_nus-3d', 'hv_ssn_regnet-400mf_secfpn_sbn-all_16xb2-2x_nus-3d', 'hv_ssn_secfpn_sbn-all_16xb2-2x_lyft-3d', 'hv_ssn_regnet-400mf_secfpn_sbn-all_16xb1-2x_lyft-3d', 'votenet_8xb16_sunrgbd-3d.py', 'votenet_8xb8_scannet-3d.py', 'votenet_iouloss_8x8_scannet-3d-18class', 'minkunet18_w16_torchsparse_8xb2-amp-15e_semantickitti', 'minkunet18_w20_torchsparse_8xb2-amp-15e_semantickitti', 'minkunet18_w32_torchsparse_8xb2-amp-15e_semantickitti', 'minkunet34_w32_minkowski_8xb2-laser-polar-mix-3x_semantickitti', 'minkunet34_w32_spconv_8xb2-amp-laser-polar-mix-3x_semantickitti', 'minkunet34_w32_spconv_8xb2-laser-polar-mix-3x_semantickitti', 'minkunet34_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti', 'minkunet34_w32_torchsparse_8xb2-laser-polar-mix-3x_semantickitti', 'minkunet34v2_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti', 'cylinder3d_4xb4-3x_semantickitti', 'cylinder3d_8xb2-laser-polar-mix-3x_semantickitti', 'pv_rcnn_8xb2-80e_kitti-3d-3class', 'fcaf3d_2xb8_scannet-3d-18class', 'fcaf3d_2xb8_sunrgbd-3d-10class', 'fcaf3d_2xb8_s3dis-3d-5class', 'spvcnn_w16_8xb2-amp-15e_semantickitti', 'spvcnn_w20_8xb2-amp-15e_semantickitti', 'spvcnn_w32_8xb2-amp-15e_semantickitti', 'spvcnn_w32_8xb2-amp-laser-polar-mix-3x_semantickitti'])
```

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
car                     0.894   0.169   0.150   0.063   0.285   0.185 
truck                   0.640   0.319   0.181   0.083   0.257   0.225 
bus                     0.769   0.332   0.185   0.058   0.472   0.257 
trailer                 0.481   0.507   0.210   0.628   0.189   0.169 
construction_vehicle    0.288   0.693   0.422   0.848   0.129   0.312 
pedestrian              0.879   0.130   0.292   0.391   0.227   0.103 
motorcycle              0.758   0.185   0.238   0.256   0.478   0.243 
bicycle                 0.625   0.155   0.257   0.319   0.208   0.009 
traffic_cone            0.790   0.122   0.327   nan     nan     nan   
barrier                 0.716   0.184   0.285   0.058   nan     nan   
11/07 23:50:24 - mmengine - INFO - Epoch(test) [6019/6019]    NuScenes metric/pred_instances_3d_NuScenes/car_AP_dist_0.5: 0.8098  NuScenes metric/pred_instances_3d_NuScenes/car_AP_dist_1.0: 0.9020  NuScenes metric/pred_instances_3d_NuScenes/car_AP_dist_2.0: 0.9274  NuScenes metric/pred_instances_3d_NuScenes/car_AP_dist_4.0: 0.9375  NuScenes metric/pred_instances_3d_NuScenes/car_trans_err: 0.1695  NuScenes metric/pred_instances_3d_NuScenes/car_scale_err: 0.1500  NuScenes metric/pred_instances_3d_NuScenes/car_orient_err: 0.0633  NuScenes metric/pred_instances_3d_NuScenes/car_vel_err: 0.2845  NuScenes metric/pred_instances_3d_NuScenes/car_attr_err: 0.1849  NuScenes metric/pred_instances_3d_NuScenes/mATE: 0.2796  NuScenes metric/pred_instances_3d_NuScenes/mASE: 0.2546  NuScenes metric/pred_instances_3d_NuScenes/mAOE: 0.3005  NuScenes metric/pred_instances_3d_NuScenes/mAVE: 0.2806  NuScenes metric/pred_instances_3d_NuScenes/mAAE: 0.1879  NuScenes metric/pred_instances_3d_NuScenes/truck_AP_dist_0.5: 0.4405  NuScenes metric/pred_instances_3d_NuScenes/truck_AP_dist_1.0: 0.6364  NuScenes metric/pred_instances_3d_NuScenes/truck_AP_dist_2.0: 0.7228  NuScenes metric/pred_instances_3d_NuScenes/truck_AP_dist_4.0: 0.7600  NuScenes metric/pred_instances_3d_NuScenes/truck_trans_err: 0.3192  NuScenes metric/pred_instances_3d_NuScenes/truck_scale_err: 0.1808  NuScenes metric/pred_instances_3d_NuScenes/truck_orient_err: 0.0827  NuScenes metric/pred_instances_3d_NuScenes/truck_vel_err: 0.2572  NuScenes metric/pred_instances_3d_NuScenes/truck_attr_err: 0.2255  NuScenes metric/pred_instances_3d_NuScenes/construction_vehicle_AP_dist_0.5: 0.0493  NuScenes metric/pred_instances_3d_NuScenes/construction_vehicle_AP_dist_1.0: 0.2080  NuScenes metric/pred_instances_3d_NuScenes/construction_vehicle_AP_dist_2.0: 0.3856  NuScenes metric/pred_instances_3d_NuScenes/construction_vehicle_AP_dist_4.0: 0.5110  NuScenes metric/pred_instances_3d_NuScenes/construction_vehicle_trans_err: 0.6927  NuScenes metric/pred_instances_3d_NuScenes/construction_vehicle_scale_err: 0.4219  NuScenes metric/pred_instances_3d_NuScenes/construction_vehicle_orient_err: 0.8481  NuScenes metric/pred_instances_3d_NuScenes/construction_vehicle_vel_err: 0.1289  NuScenes metric/pred_instances_3d_NuScenes/construction_vehicle_attr_err: 0.3117  NuScenes metric/pred_instances_3d_NuScenes/bus_AP_dist_0.5: 0.5019  NuScenes metric/pred_instances_3d_NuScenes/bus_AP_dist_1.0: 0.7763  NuScenes metric/pred_instances_3d_NuScenes/bus_AP_dist_2.0: 0.8897  NuScenes metric/pred_instances_3d_NuScenes/bus_AP_dist_4.0: 0.9091  NuScenes metric/pred_instances_3d_NuScenes/bus_trans_err: 0.3322  NuScenes metric/pred_instances_3d_NuScenes/bus_scale_err: 0.1851  NuScenes metric/pred_instances_3d_NuScenes/bus_orient_err: 0.0576  NuScenes metric/pred_instances_3d_NuScenes/bus_vel_err: 0.4719  NuScenes metric/pred_instances_3d_NuScenes/bus_attr_err: 0.2566  NuScenes metric/pred_instances_3d_NuScenes/trailer_AP_dist_0.5: 0.1682  NuScenes metric/pred_instances_3d_NuScenes/trailer_AP_dist_1.0: 0.4507  NuScenes metric/pred_instances_3d_NuScenes/trailer_AP_dist_2.0: 0.6110  NuScenes metric/pred_instances_3d_NuScenes/trailer_AP_dist_4.0: 0.6939  NuScenes metric/pred_instances_3d_NuScenes/trailer_trans_err: 0.5069  NuScenes metric/pred_instances_3d_NuScenes/trailer_scale_err: 0.2103  NuScenes metric/pred_instances_3d_NuScenes/trailer_orient_err: 0.6284  NuScenes metric/pred_instances_3d_NuScenes/trailer_vel_err: 0.1890  NuScenes metric/pred_instances_3d_NuScenes/trailer_attr_err: 0.1689  NuScenes metric/pred_instances_3d_NuScenes/barrier_AP_dist_0.5: 0.6227  NuScenes metric/pred_instances_3d_NuScenes/barrier_AP_dist_1.0: 0.7181  NuScenes metric/pred_instances_3d_NuScenes/barrier_AP_dist_2.0: 0.7549  NuScenes metric/pred_instances_3d_NuScenes/barrier_AP_dist_4.0: 0.7673  NuScenes metric/pred_instances_3d_NuScenes/barrier_trans_err: 0.1837  NuScenes metric/pred_instances_3d_NuScenes/barrier_scale_err: 0.2846  NuScenes metric/pred_instances_3d_NuScenes/barrier_orient_err: 0.0579  NuScenes metric/pred_instances_3d_NuScenes/barrier_vel_err: nan  NuScenes metric/pred_instances_3d_NuScenes/barrier_attr_err: nan  NuScenes metric/pred_instances_3d_NuScenes/motorcycle_AP_dist_0.5: 0.6544  NuScenes metric/pred_instances_3d_NuScenes/motorcycle_AP_dist_1.0: 0.7807  NuScenes metric/pred_instances_3d_NuScenes/motorcycle_AP_dist_2.0: 0.7937  NuScenes metric/pred_instances_3d_NuScenes/motorcycle_AP_dist_4.0: 0.8049  NuScenes metric/pred_instances_3d_NuScenes/motorcycle_trans_err: 0.1847  NuScenes metric/pred_instances_3d_NuScenes/motorcycle_scale_err: 0.2382  NuScenes metric/pred_instances_3d_NuScenes/motorcycle_orient_err: 0.2558  NuScenes metric/pred_instances_3d_NuScenes/motorcycle_vel_err: 0.4785  NuScenes metric/pred_instances_3d_NuScenes/motorcycle_attr_err: 0.2428  NuScenes metric/pred_instances_3d_NuScenes/bicycle_AP_dist_0.5: 0.5949  NuScenes metric/pred_instances_3d_NuScenes/bicycle_AP_dist_1.0: 0.6249  NuScenes metric/pred_instances_3d_NuScenes/bicycle_AP_dist_2.0: 0.6338  NuScenes metric/pred_instances_3d_NuScenes/bicycle_AP_dist_4.0: 0.6470  NuScenes metric/pred_instances_3d_NuScenes/bicycle_trans_err: 0.1546  NuScenes metric/pred_instances_3d_NuScenes/bicycle_scale_err: 0.2570  NuScenes metric/pred_instances_3d_NuScenes/bicycle_orient_err: 0.3193  NuScenes metric/pred_instances_3d_NuScenes/bicycle_vel_err: 0.2079  NuScenes metric/pred_instances_3d_NuScenes/bicycle_attr_err: 0.0095  NuScenes metric/pred_instances_3d_NuScenes/pedestrian_AP_dist_0.5: 0.8628  NuScenes metric/pred_instances_3d_NuScenes/pedestrian_AP_dist_1.0: 0.8736  NuScenes metric/pred_instances_3d_NuScenes/pedestrian_AP_dist_2.0: 0.8842  NuScenes metric/pred_instances_3d_NuScenes/pedestrian_AP_dist_4.0: 0.8944  NuScenes metric/pred_instances_3d_NuScenes/pedestrian_trans_err: 0.1299  NuScenes metric/pred_instances_3d_NuScenes/pedestrian_scale_err: 0.2919  NuScenes metric/pred_instances_3d_NuScenes/pedestrian_orient_err: 0.3915  NuScenes metric/pred_instances_3d_NuScenes/pedestrian_vel_err: 0.2266  NuScenes metric/pred_instances_3d_NuScenes/pedestrian_attr_err: 0.1032  NuScenes metric/pred_instances_3d_NuScenes/traffic_cone_AP_dist_0.5: 0.7675  NuScenes metric/pred_instances_3d_NuScenes/traffic_cone_AP_dist_1.0: 0.7784  NuScenes metric/pred_instances_3d_NuScenes/traffic_cone_AP_dist_2.0: 0.7948  NuScenes metric/pred_instances_3d_NuScenes/traffic_cone_AP_dist_4.0: 0.8210  NuScenes metric/pred_instances_3d_NuScenes/traffic_cone_trans_err: 0.1224  NuScenes metric/pred_instances_3d_NuScenes/traffic_cone_scale_err: 0.3265  NuScenes metric/pred_instances_3d_NuScenes/traffic_cone_orient_err: nan  NuScenes metric/pred_instances_3d_NuScenes/traffic_cone_vel_err: nan  NuScenes metric/pred_instances_3d_NuScenes/traffic_cone_attr_err: nan  NuScenes metric/pred_instances_3d_NuScenes/NDS: 0.7117  NuScenes metric/pred_instances_3d_NuScenes/mAP: 0.6841  data_time: 0.0062  time: 0.1705
```


```bash
(py310) [010796032@g21 mmdetection3d]$ cd projects/
(py310) [010796032@g21 projects]$ ln -s /data/rnd-liu/MyRepo/DeepDataMiningLearning/DeepDataMiningLearning/bevdet/ .
# make the symlink (absolute path is safest)
ln -s "$EXT" bevdet

(py310) [010796032@g21 mmdetection3d]$ # sanity check: import your module
python - <<'PY'
import sys, importlib, os
print("cwd:", os.getcwd())
print("has projects? ", any(p.endswith('/mmdetection3d') for p in sys.path))
m = importlib.import_module('projects.bevdet.bevfusion.view_transformers.cross_attn')
print('OK:', getattr(m, 'BEVCrossAttnTransform', 'no class found'))
PY
cwd: /fs/atipa/data/rnd-liu/MyRepo/mmdetection3d
has projects?  True
OK: <class 'projects.bevdet.bevfusion.view_transformers.cross_attn.BEVCrossAttnTransform'>

python tools/train.py projects/

export DEBUG_VT=1
(py310) [010796032@g20 mmdetection3d]$ python tools/train.py --config projects/bevdet/configs/bevfusion_crossattn_second_depthdistill_12e_nus-3d.py 
```