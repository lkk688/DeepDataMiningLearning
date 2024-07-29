# Development Environment for Apple Silicon

# Xcode Tools
Install Xcode command-line tools (CLT), this includes all the useful open-source tools Unix developers expect, like git, clang, and more.
```bash
xcode-select --install
```

# Brew
Install [Brew](https://brew.sh/): package manager for Macs

# Python Environment
To install conda environment in Mac, there are several options. You can install miniforge or miniconda.

## Install python via miniforge

Miniforge is a minimal installer for Conda that works well with Apple Silicon. You can download it from [miniforge](https://github.com/conda-forge/miniforge).

```bash
bash Miniforge3-latest-MacOSX-arm64.sh
#bash Miniforge3-MacOSX-x86_64.sh #if you are using Intel CPU
```

Follow the installation guide, and do not forget to restart your terminal after it is completed.

## Install python via miniconda
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

## Create Conda environment with python 
After miniconda or miniforge3 installed, you can use the `conda` command to create python environment
```bash
conda search python #check existing python versions
#remove existing environment if needed
#conda remove -n ENV_NAME --all
% conda create --name py310 python=3.10 #install python3.10
#% conda create --name py312 python=3.12 #install python3.12
% conda activate py310
% python -V
% conda install -c conda-forge pylibiio
% pip install pyadi-iio
% pip install numpy matplotlib
% conda deactivate 
```

A Minimal Setup for Data science:
```bash
conda install pandas numpy numba matplotlib seaborn scikit-learn jupyter
```


# 3. Jupyter notebook setup

