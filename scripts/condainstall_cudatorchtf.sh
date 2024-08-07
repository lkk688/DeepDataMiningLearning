# CONDA_ENV=py310cu118
# conda -V # check version
# conda info --envs
# conda create -n $CONDA_ENV python=3.10 #tensorrt==8.5.3.1 does not support python3.11
# conda deactivate
# conda activate $CONDA_ENV
conda info

# install cuda toolkit
#conda install -c conda-forge cudatoolkit=11.8.0
conda install -y cuda -c nvidia/label/cuda-11.8.0 #new method from https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#conda-installation

# Install pytorch
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# install cuDNN and Tensorflow https://www.tensorflow.org/install/source#tested_build_configurations
#python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.0
python3 -m pip install nvidia-cudnn-cu11==8.7.0.84
python3 -m pip install tensorflow[and-cuda]==2.14.0

# add cuDNN paths to env vars every time conda activate this environment
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# reload shell configs
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Verify Pytorch installation
python3 -c "import torch; print('Torch version:', torch.__version__); print(torch.cuda.is_available())"

# Verify Tensorflow installation
python3 -c "import tensorflow as tf; print('tf version:', tf.__version__); print(tf.config.list_physical_devices('GPU'))"