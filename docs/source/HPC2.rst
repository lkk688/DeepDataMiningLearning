HPC2
=====

.. _hpc2:

Load software module and request GPU node
------------------------------------------

Check available software and load the required modules in the headnode

.. code-block:: console

   $ module avail
   $ module load python39 slurm/slurm/21.08.6 gcc/11.2.0
   $ conda info --envs #check available conda environments
   $ conda activate mycondapy39
   
Use Slurm to request one GPU node, and setup required paths

.. code-block:: console

   $ srun --pty /bin/bash # request GPU node
   [sjsuid@cs002 ~]$ nvidia-smi # got the GPU node 'cs002', check the gpu via "nvidia-smi"
   [sjsuid@cs002 ~]$ module load python39 cuda12.0/toolkit/12.0.1 #load python and cuda module
   [sjsuid@cs002 ~]$ nvcc -V # check cuda version
   [sjsuid@cs002 ~]$ conda activate mycondapy39
   (mycondapy39) [sjsuid@cs002 ~]$ export LD_LIBRARY_PATH=/data/cmpe249-fa22/mycuda/TensorRT-8.4.2.4/lib:$LD_LIBRARY_PATH #add tensorrt library if needed
   (mycondapy39) [sjsuid@cs002 ~]$ CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)")) #get cudnn path
   (mycondapy39) [sjsuid@cs002 ~]$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib #add cudnn path (only needed for Tensorflow)

The GPU node does not have internet access, if you need to access the Jupyter web in your local browser, you can establish one tunnel from your local computer to the HPC headnode, then create another tunnel from the HPC headnode to the GPU node

.. code-block:: console

   $ ssh -L 10001:localhost:10001 010796032@coe-hpc2.sjsu.edu #from your local computer to HPC headnode, you can use any port number (10001)
   $ ssh -L 10001:localhost:10001 010796032@cs002 #in HPC head node to gpu node
   $ jupyter lab --no-browser --port=10001 #start the jupyter lab on port 10001 (the port should be the same port used for tunnel)

After jupyter lab is started, you can copy paste the URL shown in the terminal into your local browser to access the Jupyter lab.

Conda Environment Setup Tutorial
---------------------------------

You can install miniconda via bash or module load the available 'anaconda/3.9'. 

Install jupyter lab package in conda (make sure you are HPC headnode not the GPU node):

.. code-block:: console

   [sjsuid@coe-hpc2 ~]$ conda activate mycondapy39
   (mycondapy39) [sjsuid@coe-hpc2 ~]$ conda install -c conda-forge jupyterlab
   (mycondapy39) [sjsuid@coe-hpc2 ~]$ conda install ipykernel
   (mycondapy39) [sjsuid@coe-hpc2 ~]$ ipython kernel install --user --name=mycondapy39 #add jupyter kernel

Install Pytorch2.0 cuda11.8 version (no problem if you loaded cuda12 in GPU node)

.. code-block:: console

   (mycondapy39) [sjsuid@coe-hpc2 ~]$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia #if pytorch2.0 is not found, you can use the pip option
   (mycondapy39) [sjsuid@coe-hpc2 ~]$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -U #another option of using pip install
   (mycondapy39) [sjsuid@coe-hpc2 ~]$ python -m torch.utils.collect_env #check pytorch environment

Install cudnn (required by Tensorflow) and Tensorflow

.. code-block:: console

   (mycondapy39) [sjsuid@coe-hpc2 ~]$ python3 -m pip install nvidia-cudnn-cu11==8.6.0.163
   (mycondapy39) [sjsuid@coe-hpc2 ~]$ CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
   (mycondapy39) [sjsuid@coe-hpc2 ~]$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
   (mycondapy39) [sjsuid@coe-hpc2 ~]$ python3 -m pip install tensorflow==2.12.*

Request one GPU node, and check tensorflow GPU access

.. code-block:: console

   (mycondapy39) [sjsuid@cs002 ~]$ python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

If you see error like "RuntimeError: module compiled against API version 0xf but this version of numpy is 0xe", you can upgrade numpy version

Install other libraries

.. code-block:: console

   (mycondapy39) [sjsuid@coe-hpc2 ~]$ pip install opencv-python
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

If you want to install Numba, it conflicts with latest version of numpy (https://numba.readthedocs.io/en/stable/user/installing.html), you can uninstall numpy and install the 1.23.5 version (not too low, otherwise the SharedArray and Tensorflow will show error)

.. code-block:: console

   $ pip uninstall numpy
   $ pip install numpy==1.23.5
   $ pip install numba -U # numpy<1.24,>=1.18 is required by {'numba'}
   
You can git clone our 3D detection framework and instal the development environment

.. code-block:: console

   (mycondapy39) [sjsuid@coe-hpc2 ]$ git clone https://github.com/lkk688/3DDepth.git
   (mycondapy39) [sjsuid@coe-hpc2 3DDepth]$ python3 setup.py develop
   pip install kornia
   pip install pyquaternion
   pip install efficientnet_pytorch==0.7.0

Install pypcd

.. code-block:: console

   (mycondapy39) [010796032@coe-hpc2 3DObject]$ cd pypcd/
   (mycondapy39) [010796032@coe-hpc2 pypcd]$ python setup.py install

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

Share Conda Environment
-------------------------

You can share your environment with someone else and allow them to quickly reproduce your environment via a copy of your environment.yml file. To export environment file, ref: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html:

.. code-block:: console

   (mycondapy39) [010796032@cs004 ~]$ conda env export > mycondapy39_hpc2_environment.yml

Screen session 
---------------

Create Persistent SSH connection via screen (install screen) if you do not want to have ssh timeout. You can create one persistent session via $screen -S main_session  (if you do not perform long lasting training, do not use the persistent session, it may cause problems)
You can use $ screen -ls to view current screen session. If you want to reconnect to a previous screen session after ssh timeout, you can use

.. code-block:: console

   #'mydetector3d/tools/cfgs/nuscenes_models/bevfusion.yaml'
   #ckpt '/data/cmpe249-fa22/Mymodels/nuscenes_models/bevfusion/0522/ckpt/latest_model.pth'
   #batch size =4, gpu1, start from epoch
   [010796032@cs001 3DDepth]$ screen -ls
   There is a screen on:
         45460.main_session      (Attached)
   1 Socket in /var/run/screen/S-010796032.

   (mycondapy39) [010796032@cs001 3DDepth]$ python ./mydetector3d/tools/mytrain.py

   #'mydetector3d/tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml'
   #'/data/cmpe249-fa22/Mymodels/nuscenes_models/cbgs_pp_multihead/0522/ckpt/latest_model.pth'
   #batchsize=6, gpu2 restart from epoch 99
   $ screen -S session2
   (mycondapy39) [010796032@cs001 3DDepth]$ screen -ls
   There are screens on:
         51524.session2  (Attached)
         45460.main_session      (Attached)
   2 Sockets in /var/run/screen/S-010796032.
   (mycondapy39) [010796032@cs001 3DDepth]$ python ./mydetector3d/tools/mytrain.py

Reconnect to a previous session:

.. code-block:: console

   $ screen -r session_name #(the name you get from screen -ls)

screen -S session_name -X quit to terminate a detached session (Ctrl+a d to detach a session)
You can use Control+A, then ESC to enable the scroll up. Move up/down with the arrow keys ( ↑ and ↓ ). When you're done, hit q or Escape to get back to the end of the scroll buffer.
