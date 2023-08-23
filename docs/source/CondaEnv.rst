CondaEnv
=========

.. _CondaEnv:

Conda Environment Setup Tutorial
------------------------------------

Install Miniconda
~~~~~~~~~~~~~~~~~~

.. code-block:: console

   curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh


Create a Conda virtual environment with python 3.10 (the default python version is 3.11):

.. code-block:: console

   (base) lkk@lkk-intel13:~$ conda create --name mycondapy310 python=3.10
   (base) lkk@lkk-intel13:~$ conda activate mycondapy310 #To activate this environment
   (mycondapy310) lkk@lkk-intel13:~$ conda info --envs #check existing conda environment
   (mycondapy310) lkk@lkk-intel13:~$ conda deactivate #To deactivate an active environment

Popular packages
~~~~~~~~~~~~~~~~~

.. code-block:: console

   conda install -c conda-forge jupyterlab
   $ conda install matplotlib
   conda install -c intel scikit-learn
   conda install scikit-learn-intelex
   conda install numpy
   conda install pandas
   conda update -n base -c defaults conda
   #https://anaconda.cloud/intel-optimized-packages
   git clone https://github.com/lkk688/DeepDataMiningLearning.git
   git clone https://github.com/lkk688/WaymoObjectDetection.git
   https://github.com/lkk688/MultiModalDetector.git

Install CUDA
~~~~~~~~~~~~~~~~~~
There are two options to install CUDA: 1) using conda to install the cuda; 2) download cuda from nvidia, install cuda to the system.

Option1: install CUDA and cuDNN with conda and pip, and setup the environment path for cudnn

.. code-block:: console
   
   (mycondapy310) lkk@lkk-intel13:~$ conda install -c conda-forge cudatoolkit=11.8.0
   (mycondapy310) lkk@lkk-intel13:~$ pip install nvidia-cudnn-cu11==8.6.0.163
   (mycondapy310) lkk@lkk-intel13:~$ CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
   (mycondapy310) lkk@lkk-intel13:~$ echo $CUDNN_PATH
   /home/lkk/miniconda3/envs/mycondapy310/lib/python3.10/site-packages/nvidia/cudnn
   (mycondapy310) lkk@lkk-intel13:~$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

You can automate it with the following commands. The system paths will be automatically configured when you activate this conda environment.

.. code-block:: console
   
   (mycondapy310) lkk@lkk-intel13:~$ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
   (mycondapy310) lkk@lkk-intel13:~$ echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >>      $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
   (mycondapy310) lkk@lkk-intel13:~$ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
   (mycondapy310) lkk@lkk-intel13:~$ cat $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh #check the content of the file

Install cuda development kit, otherwise 'nvcc' is not available

.. code-block:: console

   #(mycondapy310) $ conda install -c conda-forge cudatoolkit-dev #this will install 11.7
   (mycondapy310) $ conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit #https://anaconda.org/nvidia/cuda-toolkit
   $ nvcc -V #show Cuda compilation tools

Option2: You can also go to nvidia cuda toolkit website, select the version (Ubuntu22.04 Cuda11.8) and install cuda locally

.. code-block:: console

   sudo apt install gcc
   sudo apt-get install linux-headers-$(uname -r) #The kernel headers and development packages for the currently running kernel
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
   sudo sh cuda_11.8.0_520.61.05_linux.run

When install CUDA, do not select the "install the driver" option. After cuda installation, setup the PATH and make sure that PATH includes /usr/local/cuda/bin and LD_LIBRARY_PATH includes /usr/local/cuda/lib64

.. code-block:: console

   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

You can add these path setup code in ~/.bashrc or setup in conda "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"


   
Tensorflow Installation
------------------------

Install the latest Tensorflow via pip, and verify the GPU setup

.. code-block:: console

   (mycondapy310) $ pip install tensorflow==2.12.*
   (mycondapy310) $ python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" #show [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

The tensorflow may show warning of "Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7" and "Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7" because of missing TensorRT library. You can refer the TensorRT section to install TensorRT8 and copy the libxx.so.8 to libxxx.so.7 to remove the warning.

.. code-block:: console

   $ cp /home/lkk/Developer/TensorRT-8.5.3.1/lib/libnvinfer_plugin.so.8 /home/lkk/Developer/TensorRT-8.5.3.1/lib/libnvinfer_plugin.so.7
   $ cp /home/lkk/Developer/TensorRT-8.5.3.1/lib/libnvinfer_plugin.so.8 /home/lkk/Developer/TensorRT-8.5.3.1/lib/libnvinfer_plugin.so.7

Pytorch2.0 Installation
-------------------------

.. code-block:: console

   (mycondapy310) $ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia #numpy-1.24.3 is also installed
   #torch installation may impact waymo-open-dataset, and show ModuleNotFoundError: No module named 'chardet'
   $ pip install chardet #solve the problem

Waymo OpenDataset Installation
----------------------------------

First install [openexr](https://www.excamera.com/sphinx/articles-openexr.html) for HDR images required by Waymo opendataset, then install waymo-open-dataset package

.. code-block:: console

   $ sudo apt-get install libopenexr-dev
   $ conda install -c conda-forge openexr
   $ conda install -c conda-forge openexr-python
   $ python3 -m pip install waymo-open-dataset-tf-2-11-0==1.5.1 #it will force install tensorflow2.11
   >>> from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils # test import waymo_open_dataset in python, should show no errors

3D Object Detection
-------------------------

Install the required libraries (mayavi and open3d) for 3D object visualization

.. code-block:: console

   (mycondapy310) lkk@lkk-intel13:~/Developer$ git clone https://github.com/lkk688/3DDepth.git
   (mycondapy310) $ pip install mayavi # 3D Lidar visualization: https://docs.enthought.com/mayavi/mayavi/installation.html
   (mycondapy310) $ pip install PyQt5
   (mycondapy310) $ pip install opencv-python-headless #opencv-python may conflict with mayavi
   (mycondapy310) lkk@lkk-intel13:~/Developer/3DDepth$ python ./VisUtils/testmayavi.py #test mayavi, you should see a GUI window with mayavi scene
   (mycondapy310) $ pip install open3d #install open3d: http://www.open3d.org/docs/release/getting_started.html 
   #OPEN3D upgraded the pillow, but waymo-open-dataset-tf-2-11-0 1.5.1 requires pillow==9.2.0, this warning can be ignored.
   (mycondapy310) lkk@lkk-intel13:~/Developer/3DDepth$ python ./VisUtils/testopen3d.py #test open3d
   
Install other required libraries

.. code-block:: console

   conda install -c conda-forge configargparse
   pip install -U albumentations
   pip install spconv-cu118 #check installation via import spconv
   pip install SharedArray
   pip install nuscenes-devkit

After SharedArray, test import SharedArray in python may show error of "RuntimeError: module compiled against API version 0x10 but this version of numpy is 0xe", check the current version of numpy is 1.21.5. The solution is to upgrade the numpy version, but the highest numpy version supported by numba is 1.23.5, thus we upgrade numpy

.. code-block:: console

   pip uninstall numpy
   pip install numpy==1.23.5 #no problem for import SharedArray 

After install the numpy 1.23.5, there are some errors from waymo-open-dataset, but these errors can be ignored and check the waymo-open-dataset does not show error.

.. code-block:: console

   tensorflow 2.11.0 requires protobuf<3.20,>=3.9.2, but you have protobuf 3.20.3 which is incompatible.
   waymo-open-dataset-tf-2-11-0 1.5.1 requires numpy==1.21.5, but you have numpy 1.23.5 which is incompatible.
   waymo-open-dataset-tf-2-11-0 1.5.1 requires pillow==9.2.0, but you have pillow 9.5.0 which is incompatible.

Install numba and other libraries

.. code-block:: console

   $ pip install numba
   $ pip install requests
   $ pip install --upgrade protobuf==3.19.6 #tensorflow 2.11.0 requires protobuf<3.20,>=3.9.2
   $ pip install six # required by tensorflow
   $ pip uninstall pillow
   $ pip install pillow==9.2.0 # required by waymo-open-dataset, but open3d 0.17.0 requires pillow>=9.3.0
   $ pip install tensorboardX
   $ pip install easydict
   $ pip install gpustat
   $ pip install --upgrade autopep8
   $ pip install pyyaml scikit-image onnx onnx-simplifier
   $ pip install onnxruntime
   $ pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com

You can git clone our 3D detection framework and instal the development environment

.. code-block:: console

   $ git clone https://github.com/lkk688/3DDepth.git
   (mycondapy310) lkk@lkk-intel13:~/Developer/3DDepth$ python3 setup.py develop
   nvcc fatal   : Unsupported gpu architecture 'compute_89'
   conda uninstall cudatoolkit-dev
   $ conda uninstall cudatoolkit=11.8.0
   $ conda install -c conda-forge cudatoolkit=11.8.0
   $ conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit #https://anaconda.org/nvidia/cuda-toolkit
   $ nvcc -V #show 11.8
   $ pip uninstall nvidia-cudnn-cu11 #remove cudnn8.6.0.163
   $ pip install nvidia-cudnn-cu11 #install cudnn8.9.0.131

   

TensorRT Installation
-------------------------

Use the tar installation options for [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar)
After the tar file is downloaded, untar the file, setup the TensorRT path, and install the tensorrt python package:

.. code-block:: console

   $ tar -xzvf TensorRT-8.5.3.1.Linux.x86_64-gnu.cuda-11.8.cudnn8.6.tar.gz
   $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lkk/Developer/TensorRT-8.5.3.1/lib
   (mycondapy310) lkk@lkk-intel13:~$ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lkk/Developer/TensorRT-8.5.3.1/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh #optional step, make it automatic when conda environment starts
   (mycondapy310) lkk@lkk-intel13:~/Developer/TensorRT-8.5.3.1/python$ python -m pip install tensorrt-8.5.3.1-cp310-none-linux_x86_64.whl #install the tensorrt python package
   (mycondapy310) lkk@lkk-intel13:~/Developer/TensorRT-8.5.3.1/graphsurgeon$ python -m pip install graphsurgeon-0.4.6-py2.py3-none-any.whl
   (mycondapy310) lkk@lkk-intel13:~/Developer/TensorRT-8.5.3.1/onnx_graphsurgeon$ python -m pip install onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
   
Check the TensorRT sample code from [TensorRTSample](https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html#samples)

huggingface Tutorial
--------------------
https://github.com/huggingface/transformers

https://huggingface.co/docs/transformers/index

https://huggingface.co/learn/nlp-course/chapter1/2?fw=pt