# Setup in Intel13Rack
```bash
lkk@lkk-intel13rack:~$ curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
lkk@lkk-intel13rack:~$ python3 -V
Python 3.10.12
lkk@lkk-intel13rack:~$ bash Miniconda3-latest-Linux-x86_64.sh -b -u
lkk@lkk-intel13rack:~$ source ~/miniconda3/bin/activate
(base) lkk@lkk-intel13rack:~$ conda init bash
(base) lkk@lkk-intel13rack:~$ gcc --version
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
```

update GCC11 to GCC13
```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-13 g++-13
gcc --version
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 13
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 13
gcc --version
gcc (Ubuntu 13.1.0-8ubuntu1~22.04) 13.1.0
```

Install Cuda 12.6
```bash
(base) lkk@lkk-intel13rack:~/MyRepo/DeepDataMiningLearning$ cd scripts/
chmod +x cuda_local_install.sh
bash cuda_local_install.sh 126 ~/nvidia 1
source ~/.bashrc
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Fri_Jun_14_16:34:21_PDT_2024
Cuda compilation tools, release 12.6, V12.6.20
Build cuda_12.6.r12.6/compiler.34431801_0
$ which nvcc
/home/lkk/nvidia/cuda-12.6/bin/nvcc
$ sudo mkdir -p /usr/local/cuda/bin
$ sudo ln -s /home/lkk/nvidia/cuda-12.6/bin/nvcc /usr/local/cuda/bin/nvcc
export CPATH=$CPATH:/home/lkk/nvidia/cuda-12.6/include
#test cuda
(base) lkk@lkk-intel13rack:~/nvidia$ nvcc testcuda.cu -o testcuda
(base) lkk@lkk-intel13rack:~/nvidia$ ./testcuda
#the following cuda sample has problems
(base) lkk@lkk-intel13rack:~/nvidia$ git clone https://github.com/NVIDIA/cuda-samples.git
(base) lkk@lkk-intel13rack:~/nvidia/cuda-samples$ mkdir build && cd build
sudo apt  install cmake
(base) lkk@lkk-intel13rack:~/nvidia/cuda-samples/build$ cmake ..
cd ~/nvidia/cuda/samples/1_Utilities/deviceQuery
make
#find / -name deviceQuery
./deviceQuery
```

Create Conda virtual environment
```bash
conda create --name py312 python=3.12
conda activate py312
conda info --envs #check existing conda environment
% conda env list
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Install Squid:
```bash
(base) lkk@lkk-intel13rack:~$ sudo apt install squid
sudo cp /etc/squid/squid.conf /etc/squid/squid.conf.bak
sudo nano /etc/squid/squid.conf
#add the following
    acl localnet src 172.16.1.0/24
    http_access allow localnet
$ sudo systemctl restart squid
#Enable squid to start on boot:
sudo systemctl enable squid
#configure internal server
export http_proxy="http://10.31.81.70:3128"
export https_proxy="http://10.31.81.70:3128"
```

# New Conda Setup in P100
```bash
conda create --name py312 python=3.12
conda activate py312
conda info --envs #check existing conda environment
$ conda install cuda -c nvidia/label/cuda-12.4 
#install pytorch 2.6 CUDA12.4
pip3 install torch torchvision torchaudio
```

# New Conda Setup in WSL of Alienware 3090
```bash
conda create --name py312 python=3.12
conda activate py312
conda info --envs #check existing conda environment
$ conda install cuda -c nvidia/label/cuda-12.6
#install pytorch 2.6 CUDA12.6
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install matplotlib
pip install opencv-python
```

```bash
$ conda create --name py310 python=3.10 -y
conda activate py310
(py310) lkk688@newalienware:~/Developer/DeepDataMiningLearning$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
$ pip3 install torch torchvision
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
mim install 'mmdet>=3.0.0'
mim uninstall 'mmdet>=3.0.0'
mim install mmdet
$ git clone https://github.com/open-mmlab/mmdetection3d.git
(py310) lkk688@newalienware:~/Developer/mmdetection3d$ pip install -v -e .
$ mim install "mmcv==2.1.0"

#new setup
pip install -U openmim
mim install mmengine
mim install 'mmcv<2.2.0,>=2.0.0rc4'
Successfully installed mmcv-2.1.0
mim install 'mmdet>=3.0.0'
(py310) lkk688@newalienware:~/Developer/mmdetection3d$ pip install -v -e .
>>> import mmdet3d
>>> print(mmdet3d.__version__)
1.4.0
>>> import mmdet
>>> print(mmdet.__version__)
3.3.0
pip uninstall numpy
pip install 'numpy<2'
```

# 5090 Setup
Install the recommended driver: This is the best approach as ubuntu-drivers automatically uses DKMS to build the module and handle the MOK enrollment process.
```bash
sudo ubuntu-drivers devices
sudo ubuntu-drivers install nvidia:580-open
```