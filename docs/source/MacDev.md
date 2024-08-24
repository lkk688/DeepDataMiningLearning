# Development Environment for Apple Silicon

## Xcode Tools
Install Xcode command-line tools (CLT), this includes all the useful open-source tools Unix developers expect, like git, clang, and more.
```bash
xcode-select --install
```

## Brew
Install [Brew](https://brew.sh/): package manager for Macs

## Python Environment
To install conda environment in Mac, there are several options. You can install miniforge or miniconda.

### Install python via miniforge

Miniforge is a minimal installer for Conda that works well with Apple Silicon. You can download it from [miniforge](https://github.com/conda-forge/miniforge).

```bash
bash Miniforge3-latest-MacOSX-arm64.sh
#bash Miniforge3-MacOSX-x86_64.sh #if you are using Intel CPU
```

Follow the installation guide, and do not forget to restart your terminal after it is completed.

### Install python via miniconda
Download miniconda from [link](https://docs.anaconda.com/miniconda/)

```bash
% curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o Miniconda3-latest-MacOSX-x86_64.sh
% bash Miniconda3-latest-MacOSX-x86_64.sh -b -u
lkk@kaikais-mbp2019 Developers % source ~/miniconda3/bin/activate
(base) lkk@kaikais-mbp2019 Developers % conda init bash
```
Restart your terminal after it is completed.

In Mac, if you face `md5sum: command not found` problem, install it via `brew`
```bash
brew update
brew upgrade
brew install md5sha1sum
```

### Create Conda environment with python 
After miniconda or miniforge3 installed, you can use the `conda` command to create python environment
```bash
conda search python #check existing python versions
#remove existing environment if needed
#conda remove -n ENV_NAME --all
% conda create --name py310 python=3.10 #install python3.10
#% conda create --name py312 python=3.12 #install python3.12
% conda activate py310
% python -V
% conda deactivate 
conda info --envs #check existing conda environment
```

A Minimal Setup for Data science:
```bash
conda install pandas numpy numba matplotlib seaborn scikit-learn jupyter
conda install -y Pillow scipy pyyaml scikit-image 
pip install opencv-python-headless
pip install PyQt5
pip install pyqtgraph
pip install pyqt6
pip install pyside6 #Side6 and QT5 works in Mac
```

Setup for ADI IIO:
```bash
(py312) kaikailiu@kaikais-mbp Documents % conda install -c conda-forge pylibiio
Channels:
 - conda-forge
 - defaults
Platform: osx-arm64
% pip install pyadi-iio
pip install pyqt5 pyqt6 PySide6 pyqtgraph opencv-python-headless PyOpenGL PyOpenGL_accelerate pyopengl
pip install sionna DeepMIMO
(mypy310) (base) kaikailiu@Kaikais-MBP radarsensing % pip install tensorflow==2.14.0
#Test tensorflow
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
pip install sionna
```

### Jupyter notebook setup
```bash
conda install -y -c conda-forge jupyterlab
conda install -y ipykernel
jupyter kernelspec list #view current jupyter kernels
ipython kernel install --user --name=py312
```

## Network
Install zrok tunnel:
```bash
kaikailiu@Kaikais-MacBook-Pro Downloads % mkdir -p /tmp/zrok && tar -xf ./zrok_0.4.39_darwin_arm64.tar.gz -C /tmp/zrok
% cd /tmp/zrok
% mkdir -p bin && install /tmp/zrok/zrok bin/
#nano ~/.zshrc
% export PATH=/tmp/zrok/bin:$PATH
% zrok version
% zrok enable DmXXX #add environment
ssh lkk@127.0.0.1 -p 9191
```

If you want to ssh into a remote linux machine via ssh, enter share in the Linux machine
```bash
$ zrok share private --backend-mode tcpTunnel 192.168.9.1:22
#it will show access your share with: zrok access private gdjf3oz1pudh
```

In your local Mac, enter `zrok access private gdjf3oz1pudh`, it will show `tcp://127.0.0.1:9191 -> gdjf3oz1pudh`. In another Mac terminal, ssh into the remote machine
```bash
% ssh lkk@127.0.0.1 -p 9191
# same to this: ssh lkk@localhost -p 9191
```
