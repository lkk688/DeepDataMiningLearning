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
module load python3/3.11.5 cuda/11.8 anaconda/3.9 slurm-22-05-7-1-gcc-12.2.0-kzyx6rx
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
srun -p gpu --gres=gpu --pty /bin/bash
nvidia-smi
conda activate mycondapy311
exit
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

### mmdetection3d
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