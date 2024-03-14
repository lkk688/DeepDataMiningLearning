HPC
=====

.. _hpc:

Author:
   * *Kaikai Liu*, Associate Professor, SJSU
   * **Email**: kaikai.liu@sjsu.edu
   * **Web**: http://www.sjsu.edu/cmpe/faculty/tenure-line/kaikai-liu.php


Introduction of HPC
--------------------
You can access our coe HPC access information from the main website: http://coe-hpc-web.sjsu.edu.
   * A total of 36 nodes, 15 include 1 NVIDIA Tesla P100 12 GB GPUs, and 1 has 2 NVIDIA Tesla P100 GPUs 20 nodes (compute nodes) have 128 GB of RAM, and 16 nodes (GPU and condo nodes) feature 256 GB.
   * The HPC has 110 TB of home directory and data storage, which is available from all nodes via /home and /data. Additionally, the HPC has a high-throughput Lustre parallel file system with a usable space of 0.541 PB, available via /scratch. Each group will have a sub-directory in /data and /scratch that they can write to.

If you have provided your SJSU ID to your instructor, you can access the HPC using your SJSU account by using the following command: "ssh SJSUID@coe-hpc1.sjsu.edu" (replace the SJSUID with your own SJSU ID number).
   * The accessible group folders are "/data/cmpe258-sp24" and "/scratch/cmpe258-sp24" with "cmpe258-sp24" representing your class ID. Please create sub-directories within one of these group directories for your project, naming them after your name or your group. Your dataset should be placed in "/data/cmpe249-fa23." Put your code, python environment, and other temporary data into "/scratch/cmpe258-sp24".
   * You will be provided access to your private home directory on the head node. However, please note that **the head node is not intended for heavy computation or extensive data storage**. Do not store datasets, trained models, and other large files in your home directory. Our HPC admin enforces the home directory storage not exceeding 20GB. If you require substantial computation, you can request a GPU/CPU node.
   * Your requested GPU/CPU node is internally connected to your head node, and it does not have internet access when you log in (can be connected to internet via proxy). Therefore, ensure that you download data or install any necessary software on the HPC host machine, not the GPU node.
   * It's important to remember that the HPC system is provided as a courtesy, and there is no guarantee of computing resources.
   * Some frameworks will download the datasets and models into their default cache directories (i.e., in your home directory). Please do the following changes to setup a different cache folder for pytorch and huggingface. Add the following lines into your "~/.bashrc" file:

.. code-block:: console

   nano ~/.bashrc
   #add the following lines
   export HF_HOME=/data/cmpe258-sp24/.cache/huggingface
   export TORCH_HOME=/data/cmpe258-sp24/.cache/torch
   source ~/.bashrc #to take effect

Ref our previous HPC tutorial: https://docs.google.com/document/d/1bNOUUqkeb9ItTsGHAXHLvFsLR6fA0MeZcBRVMCnvtms/edit?usp=sharing



Remote access
-------------

SSH Access
~~~~~~~~~~
To SSH into the HPC host machine, you need to establish a VPN connection to the campus first. You can do so by visiting this link: https://www.sjsu.edu/it/services/network/vpn/index.php. Your SJSUID should have already been granted access to the HPC through your instructor or project advisor. If you encounter any issues with access, please request it through your advisor, replacing "SJSUID" with your own SJSU ID. Please refrain from reaching out to the campus IT department regarding HPC access.

.. code-block:: console

   $ ssh SJSUID@coe-hpc1.sjsu.edu #E.g., ssh 0107960xx@coe-hpc.sjsu.edu
   # password is the same to your SJSU account

File Transfer
~~~~~~~~~~~~~~~~~

You have several options for transferring files to and from HPC:
   * You can employ the "scp" command for secure file copy.
   * The built-in file browser within JupyterLab allows you to upload or download files.
   * Consider using "sshfs" to mount the HPC data folder onto your local system for easy access. In your **local** linux system:

.. code-block:: console

   $ sudo apt-get install sshfs
   $ sshfs xxx@coe-hpc1.sjsu.edu:/data/cmpeXXX  .

VSCode Remote to HPC
~~~~~~~~~~~~~~~~~~~~~

You can also make use of Visual Studio Code Remote Debugging (https://code.visualstudio.com/docs/remote/ssh) with the following steps:
   * Install the "Remote - SSH" extension in Visual Studio Code.
   * Add a new SSH connection in the "Remote" extension, then select and open the "coe-hpc1" server.
   * Open the desired folder as your working directory.
   * You can also open Jupyter notebook files in VSCode and use Git to synchronize with GitHub. Please note that this setup is linked only to the head node and not the compute node (GPU node).
   * Visual Studio Code has been successfully tested on the GPU node and is fully functional in HPC2 (please note that it remains non-functional in HPC1). This allows for seamless step-by-step debugging. In order to activate this feature, it's essential to enable dual-hop SSH. It's important to mention that Git sync isn't available on the GPU node due to its lack of internet access. To facilitate this, please add the following configurations to your ".ssh/config" file (can be opened via VSCode Remote Explorer and click the setting button).

.. code-block:: console

   Host coe-hpc1
      HostName coe-hpc1.sjsu.edu
      ForwardAgent yes
      User 0107xxxxxx #your SJSUID
      ForwardX11 yes
      ForwardX11Trusted yes
      ServerAliveInterval 30

VSCode Remote to Lab's Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dual-hop SSH to the HPC1 GPU node isn't operational. An alternative method involves using dual-hop SSH to Prof. Liu's lab server, which utilizes the same NVIDIA P100 GPU and is named "hpc1p100." The lab's P100 server is located in ENG276, possesses internet access, and functions as a conventional Linux server without any HPC-related limitations. You are welcome to utilize the lab's P100 server for the purposes of debugging and testing your code. If you have a long-running training job, we recommend that you submit a request for a GPU node from HPC1. 

To access the lab's server, add the following to your ".ssh/config" file

.. code-block:: console

   Host hpc1p100
     Hostname 130.65.157.216
     User student
     ForwardX11 yes
     ForwardX11Trusted yes
     ServerAliveInterval 30
     ProxyCommand ssh coe-hpc1 -W %h:%p

You can use the following command to access the lab's server

.. code-block:: console

   $ ssh hpc1p100
   (xxx@coe-hpc1.sjsu.edu) Password: #your HPC headnode password
   student@130.65.157.216's password: #lab server's password

You will require your SJSU password for the initial authentication to the HPC1 headnode, followed by a secondary password for the lab's P100 machine (the account name is "student," and you should request the password from Prof. Liu). 

Inside the lab server, you can access some existing data or put your own data into this folder "/DATA5T2/Datasets/". 
You can also mount the HPC folder into the local server into the folder of "/data/cmpe249-fa23", I alreay used "sudo chmod 777 /data/cmpe249-fa23" to add write permission. You can use sshfs to mount the HPC folder:

.. code-block:: console

   (base) student@p100:~$ sshfs yousjsuid@coe-hpc1.sjsu.edu:/data/cmpe249-fa23 /data/cmpe249-fa23
   (base) student@p100:~$ ls /data/cmpe249-fa23


GPU node internet access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   [010796032@g9 ~]$ export http_proxy=http://172.16.1.2:3128
   [010796032@g9 ~]$ export https_proxy=http://172.16.1.2:3128
   #you can also put these into ~/.bashrc
   curl --proxy http://172.16.1.2:3128 "https://www.sjsu.edu"
   # git works
   curl "https://www.sjsu.edu" #also works
  


X11 Window forwarding
~~~~~~~~~~~~~~~~~~~~~
X11 Forwarding gives you the ability to run GUIs from HPC on your own local machine. X11 window forwarding is also tested and working fine for Matplotlib and OpenCV (both terminal and VSCode)
   * For Macs, your best option is to download xQuartz from xQuartz.org. This is free software which will allow you to forward X11 on a Mac. Download the xQuartz DMG, open it, and follow the installation instructions.
   * For Linux, depending on your distribution, there may be no pre-requisites.
   * For Windows, you can use MobaXterm (https://mobaxterm.mobatek.net/download-home-edition.html) for all your Windows X11 Forwarding needs. Run MobaXterm and use the Start local terminal button to begin a session. 
   * You can also use Putty with Xming (https://sourceforge.net/projects/xming/) in Windows. Launch Xming: A small program will appear in the taskbar; keep this running for the duration of the session. Launch PuTTy, In the left-hand menu, expand “SSH”, open the “X11” menu, and check “Enable X11 Forwarding.” Go back to the “Session” menu, and under “Host Name” type HPC server address "SJSUID@coe-hpc1.sjsu.edu", then press Open.
   * After your local machine setup is finished, ssh to the HPC server via "-Y" option: "ssh -Y 010xx@coe-hpc1.sjsu.edu"

Load software module and request GPU node
------------------------------------------

Check available software via "module avail" and load the required modules in the headnode

.. code-block:: console

   $ module avail
   $ module load python3/3.11.5 cuda/11.8 anaconda/3.9 slurm-22-05-7-1-gcc-12.2.0-kzyx6rx

You can check and activate your conda environments (check Conda installation section if your conda is not installed)

.. code-block:: console

   $ conda info --envs #check available conda environments
   $ conda activate mycondapy311


Use Slurm to request one CPU/GPU node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To request CPU node and get the interactive bash, we can use Slurm (srun) on the host machine: 

.. code-block:: console

   [0107xxx@coe-hpc1 ~]$ srun --pty /bin/bash
   [0107xxx@c4 ~]$ 
   [0107xxx@c4 ~]$ exit # exit the computing node if you are not used

To request GPU node and get the interactive bash, we need to use srun to request one GPU node (g3 is your allocated node)

.. code-block:: console

   [0107xxx@coe-hpc1 ~]$ srun -p gpu --gres=gpu --pty /bin/bash
   [0107xxx@g3 ~]$ nvidia-smi #check GPU info
   [0107xxx@g3 ~]$ conda activate mycondapy311 #activate conda environment
   [0107xxx@g3 ~]$ exit # exit the GPU node if you are not used

.. note::
   If you see srun: job 26773 queued and waiting for resources, that means there is no available GPUs for you to use in HPC, you need to wait until you see: srun: job 26773 has been allocated resources. You will be automatically log into the allocated GPU

If you want to load the TensorRT library (optional):

.. code-block:: console

   [sjsuid@cs002 ~]$ conda activate mycondapy311
   (mycondapy10) [sjsuid@cs002 ~]$ export LD_LIBRARY_PATH=/data/cmpe258-sp24/mycuda/TensorRT-8.4.2.4/lib:$LD_LIBRARY_PATH #add tensorrt library if needed


Jupyterlab access
~~~~~~~~~~~~~~~~~

The GPU node does not have internet access. If you wish to access the Jupyter web interface in your local browser, you can set up a tunnel from your local computer to the HPC headnode and then create another tunnel from the HPC headnode to the GPU node (**change the port number 10001 to other numbers**).

.. code-block:: console

   $ ssh -L 10001:localhost:10001 0107xxx@coe-hpc1.sjsu.edu #from your local computer to HPC headnode, forwards any connection to port 10001 on the local machine to port 10001 on localhost
   $ ssh -L 10001:localhost:10001 0107xxx@g7 #in HPC head node to gpu node
   #activate python virtual environment, e.g., conda activate xxx
   $ jupyter lab --no-browser --port=10001 #start the jupyter lab on port 10001 (the port should be the same port used for tunnel)

After jupyter lab is started, you can copy paste the URL shown in the terminal into your local browser to access the Jupyter lab.

.. note::
   Change the port number 10001 to other numbers. If you found the jupyter creates a different port number, it may means your previous port is occupied and you cannot access your notebook via the previous port number.

Conda Environment Setup Tutorial
---------------------------------

You can install miniconda via bash or **module load the available 'anaconda/3.9'**. 

If you want to install the latest version of miniconda, you can download Miniconda3 latest version via curl and run the install script

.. code-block:: console

   $ curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
   $ bash Miniconda3-latest-Linux-x86_64.sh
   installation finished.
   Do you wish the installer to initialize Miniconda3
   by running conda init? [yes|no]
   modified      /home/010796032/.bashrc

   ==> For changes to take effect, close and re-open your current shell. <==

   If you'd prefer that conda's base environment not be activated on startup, 
      set the auto_activate_base parameter to false: 
   $ source ~/.bashrc #Take effect via source bashrc
   $ conda -V # check version
   $ conda info --envs #Check available conda environments

You can create a new conda virtual environment

.. code-block:: console

   $ conda create --name mycondapy311 #python=3.11 #add python means you want to install a new python inside the conda
   # To activate this environment, use
   #
   #     $ conda activate mycondapy311
   #
   # To deactivate an active environment, use
   #
   #     $ conda deactivate


Install jupyter lab package in conda (make sure you are HPC headnode not the GPU node):

.. code-block:: console

   [sjsuid@coe-hpc ~]$ conda activate mycondapy311
   [sjsuid@coe-hpc ~]$ conda install -c conda-forge jupyterlab
   [sjsuid@coe-hpc ~]$ conda install ipykernel
   $ jupyter kernelspec list #view current jupyter kernels
   [sjsuid@coe-hpc ~]$ ipython kernel install --user --name=mycondapy311 #add jupyter kernel

CUDA Setup Tutorial
---------------------------------
There are multiple options to install cuda in HPC: 1) module load the preinstalled cuda version (recommended); 2) install one cuda version inside the conda; 3) install cuda into your user directory outside of the conda (not recommended).

Option1: module load the preinstalled cuda version
===================================================

If you module load the cuda 11.8 via the follow script, you should be able to access the cuda in the GPU node. You can use "nvcc -V" to the cuda version

.. code-block:: console

   module load python3/3.11.5 cuda/11.8
   [@g8 cmpe258-sp24]$ nvcc -V
   nvcc: NVIDIA (R) Cuda compiler driver
   Copyright (c) 2005-2022 NVIDIA Corporation
   Built on Wed_Sep_21_10:33:58_PDT_2022
   Cuda compilation tools, release 11.8, V11.8.89
   Build cuda_11.8.r11.8/compiler.31833905_0

Option2: Install CUDA 11.8 under Conda
======================================
In order to install cuda under conda, you need to activate the conda virtual environment first, and install the cudatoolkit:

.. code-block:: console

   (mycondapy311) [sjsuid@coe-hpc ~]$ conda install -c conda-forge cudatoolkit=11.8.0

Install cuda development kit, otherwise 'nvcc' is not available in GPU node (This step is optional if you do not need cuda compiler)

.. code-block:: console

   (mycondapy311) [sjsuid@coe-hpc ~]$ conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit #https://anaconda.org/nvidia/cuda-toolkit
   $ nvcc -V #show Cuda compilation tools in GPU node

Pytorch installation
---------------------

Install Pytorch2.x cuda11.8 version (no problem if you loaded cuda12 in GPU node)

.. code-block:: console

   (mycondapy311) [sjsuid@coe-hpc ~]$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia #if pytorch2.0 is not found, you can use the pip option
   (mycondapy311) [sjsuid@coe-hpc ~]$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -U #another option of using pip install
   (mycondapy311) [sjsuid@coe-hpc ~]$ python -m torch.utils.collect_env #check pytorch environment

Install cudnn (required by Tensorflow) and Tensorflow via pip: https://www.tensorflow.org/install/pip

.. code-block:: console

   (mycondapy311) [sjsuid@coe-hpc ~]$ python3 -m pip install nvidia-cudnn-cu11==8.6.0.163
   (mycondapy311) [sjsuid@coe-hpc ~]$ CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
   (mycondapy311) [sjsuid@coe-hpc ~]$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
   (mycondapy311) [sjsuid@coe-hpc ~]$ python3 -m pip install tensorflow==2.13.*

Request one GPU node, and check tensorflow GPU access

.. code-block:: console

   (mycondapy311) [sjsuid@cs002 ~]$ python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

If you see error like "RuntimeError: module compiled against API version 0xf but this version of numpy is 0xe", you can upgrade numpy version

Install other libraries
------------------------

.. code-block:: console

   (mycondapy311) [sjsuid@coe-hpc2 ~]$ pip install opencv-python
   pip install configargparse
   pip install -U albumentations
   pip install spconv-cu118
   pip install SharedArray
   pip install tensorboardX
   pip install easydict
   pip install gpustat
   pip install --upgrade autopep8
   pip install pyyaml scikit-image onnx onnx-simplifier
   pip install onnxruntime
   pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com
   pip install waymo-open-dataset-tf-2-6-0
   pip install --upgrade protobuf==3.20.0 #waymo-open-dataset does not support higher version of protobuf
   pip install nuscenes-devkit
   pip install onnx

If you want to install Numba, it conflicts with latest version of numpy (https://numba.readthedocs.io/en/stable/user/installing.html), you can uninstall numpy and install the 1.23.5 version (not too low, otherwise the SharedArray and Tensorflow will show error)

.. code-block:: console

   $ pip uninstall numpy
   $ pip install numpy==1.23.5
   $ pip install numba -U # numpy<1.24,>=1.18 is required by {'numba'}
   
You can git clone our 3D detection framework and instal the development environment

.. code-block:: console

   (mycondapy311) [sjsuid@coe-hpc2 ]$ git clone https://github.com/lkk688/3DDepth.git
   (mycondapy311) [sjsuid@coe-hpc2 3DDepth]$ python3 setup.py develop
   pip install kornia
   pip install pyquaternion
   pip install efficientnet_pytorch==0.7.0

Install pypcd

.. code-block:: console

   (mycondapy311) [010796032@coe-hpc2 3DObject]$ cd pypcd/
   (mycondapy311) [010796032@coe-hpc2 pypcd]$ python setup.py install

Install Huggingface

.. code-block:: console

   (mycondapy39) [010796032@coe-hpc2 DeepDataMiningLearning]$ pip install transformers
   (mycondapy39) [010796032@coe-hpc2 DeepDataMiningLearning]$ pip install datasets
   (mycondapy39) [010796032@coe-hpc2 DeepDataMiningLearning]$ pip install sentencepiece
   (mycondapy39) [010796032@coe-hpc2 DeepDataMiningLearning]$ pip install scikit-learn
   (mycondapy39) [010796032@coe-hpc2 DeepDataMiningLearning]$ pip install accelerate
   (mycondapy39) [010796032@coe-hpc2 DeepDataMiningLearning]$ pip install evaluate
   (mycondapy39) [010796032@coe-hpc2 DeepDataMiningLearning]$ pip install xformers #it will change torch2.0.0+cu118 to (2.0.1+cu117), change nvidia-cublas-cu11 and nvidia-cudnn-cu11
   (mycondapy39) [010796032@coe-hpc2 DeepDataMiningLearning]$ pip install umap-learn

New conda environment based on Python3.10: mycondapy310

.. code-block:: console

   $ conda create --name mycondapy310 python=3.10
   conda activate mycondapy310
   (mycondapy310) [010796032@coe-hpc1 DeepDataMiningLearning]$ python -V
   Python 3.10.11
   $ conda install -c conda-forge cudatoolkit=11.8.0
   $ conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
   $ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   $ conda install matplotlib
   $ pip install torchtext
   $ pip install portalocker #required by torchtext
   $ conda install -c conda-forge spacy #https://spacy.io/usage
   $ conda install -c conda-forge cupy #https://docs.cupy.dev/en/stable/install.html
   $ python -m spacy download en_core_web_sm
   >>> import spacy
   >>> spacy.prefer_gpu()
   True
   >>> nlp = spacy.load("en_core_web_sm")
   $ pip install configargparse
   $ pip install datasets
   $ conda install -c conda-forge scikit-learn
   $ pip install albumentations #call scipy, cause  version `GLIBCXX_3.4.30' not found
   $ conda install -c conda-forge gcc=12.1.0 #solve the `GLIBCXX_3.4.30' problem
   $ conda install -c conda-forge tensorboard
   (mycondapy310) [010796032@g4 MultiModalClassifier]$ python setup.py develop
   pip install -q torchinfo
   $ conda install -c conda-forge jupyterlab
   ipython kernel install --user --name=mycondapy310
   pip install pyyaml scikit-image onnx onnx-simplifier
   pip install onnxruntime
   pip install seaborn
   pip install sacrebleu
   pip install sacremoses
   pip install nltk
   pip install rouge_score
   pip install sentencepiece
   pip install protobuf

Install MMdetection3d:

.. code-block:: console

   pip install -U openmim
   mim install mmengine
   mim install 'mmcv>=2.0.0rc4'
   mim install 'mmdet>=3.0.0'
   (mycondapy310) [010796032@cs001 MyRepo]$ git clone https://github.com/open-mmlab/mmdetection3d.git
   (mycondapy310) [010796032@cs001 MyRepo]$ cd mmdetection3d/
   nano requirements/runtime.txt #remove open3d in the list
   (mycondapy310) [010796032@cs001 mmdetection3d]$ pip install -v -e .
   pip install cumm-cu118
   pip install spconv-cu118

Test code:

.. code-block:: console

   (mycondapy310) [010796032@g5 nlp]$ python torchtransformer.py
   | epoch   3 |  2800/ 2928 batches | lr 4.51 | ms/batch 11.77 | loss  2.30 | ppl     9.94
   -----------------------------------------------------------------------------------------
   | end of epoch   3 | time: 36.15s | valid loss  1.03 | valid ppl     2.79
   -----------------------------------------------------------------------------------------
   =========================================================================================
   | End of training | test loss  0.98 | test ppl     2.68
   =========================================================================================

Container
----------
Load Singularity to use container: 

.. code-block:: console

   [010796032@coe-hpc1 cmpe249-fa23]$ module load singularity/3.10.3

You can run the container in CPU or GPU node and mount the data folder (your home folder is mounted by default):

.. code-block:: console

   [010796032@g5 cmpe249-fa23]$ singularity run --bind /data/cmpe249-fa23:/data/ --nv --writable myros2humblecuda117/
   Singularity> cat /etc/os-release
   PRETTY_NAME="Ubuntu 22.04.2 LTS"
   Singularity> ls /data/
   COCOoriginal      Waymo200  kitti                myros2humblecuda117.tar  torchhome
   Huggingfacecache  coco      myros2humblecuda117  nuScenes                 torchvisiondata
   Singularity> python
   Python 3.10.12 (main, Jun 11 2023, 05:26:28) [GCC 11.4.0] on linux
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import torch
   >>> torch.cuda.is_available()
   True
   >>> torch.cuda.device_count()
   1
   >>> torch.cuda.get_device_name(0)
   'Tesla P100-PCIE-12GB'

Run pytorch test script:

.. code-block:: console

   Singularity> pwd
   /home/010796032/MyRepo/DeepDataMiningLearning/DeepDataMiningLearning
   Singularity> python singleGPU.py
   Using cuda device
   Shape of X [N, C, H, W]: torch.Size([32, 1, 28, 28])
   Shape of y: torch.Size([32]) torch.int64
   [GPUcuda] Epoch 0 | Batchsize: 32 | Steps: 1875
   Singularity> python siamese_network.py
   Train Epoch: 14 [59520/60000 (99%)]     Loss: 0.000155
   Test set: Average loss: 0.0000, Accuracy: 9959/10000 (100%)


Test ROS2:

.. code-block:: console

   Singularity> printenv | grep -i ROS
   SINGULARITY_NAME=myros2humblecuda117
   SINGULARITY_CONTAINER=/data/cmpe249-fa23/myros2humblecuda117
   ROS_ROOT=/opt/ros/humble
   ROS_DISTRO=humble
   Singularity> echo ${ROS_DISTRO}
   humble
   Singularity> source /opt/ros/${ROS_DISTRO}/setup.bash
   Singularity> rosdep update
   Singularity> ros2 run demo_nodes_cpp talker
   [INFO] [1694195932.574826844] [talker]: Publishing: 'Hello World: 1'
   [INFO] [1694195933.574802426] [talker]: Publishing: 'Hello World: 2'
   [INFO] [1694195934.574829172] [talker]: Publishing: 'Hello World: 3'
   [INFO] [1694195935.574795028] [talker]: Publishing: 'Hello World: 4'

Exit the container:

.. code-block:: console
   Singularity> pip install pypdf
   Singularity> exit
