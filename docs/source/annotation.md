
# Label Studio
Label Studio is an open-source data labeling tool. It's designed to help you label various types of data (images, text, audio, etc.) for machine learning model training.

https://labelstud.io/guide/install

Poetry helps you declare, manage, and install the libraries your Python project depends on. It ensures you have the correct versions of those libraries. It replaces older methods that could be cumbersome, like managing requirements.txt files. Poetry also assists in packaging your Python project for distribution. This means it helps you create packages that others can easily install and use.

Uses pyproject.toml for project configuration. The poetry install command is crucial for setting up your project's environment.
* Generates a poetry.lock file to ensure consistent installations across different environments. 
* The poetry.lock file plays a crucial role here. It stores the exact versions of all installed packages, ensuring consistent installations across different environments.


A fundamental pyproject.toml file will typically include these sections:
* [tool.poetry]: This section contains Poetry-specific settings, such as your project's name, version, and dependencies.
* [build-system]: This section defines the build system requirements.
* [project]: This section contains core metadata about your project.

`https://github.com/HumanSignal/label-studio/blob/develop/label_studio/__init__.py`: This file signifies that the label_studio directory is a Python package. 
* In Python, the `__init__.py` file serves two primary purposes: 1) Package Marker: Its presence in a directory tells Python that the directory should be treated as a Python package; 2) It can be used to execute initialization code when the package is imported. It can also be used to control what is imported when someone uses `from xx import *`.
* `importlib.metadata` is a module in the Python standard library (since Python 3.8) that provides tools for accessing metadata about installed Python packages. It allows you to retrieve information like the package's version, dependencies, entry points


```bash
git clone https://github.com/HumanSignal/label-studio.git

# install dependencies
cd label-studio
pip install poetry
poetry install

# run db migrations
# poetry run tells Poetry to execute a command within the project's virtual environment.
poetry run python label_studio/manage.py migrate

# collect static files
poetry run python label_studio/manage.py collectstatic

# start the server in development mode at http://localhost:8080
poetry run python label_studio/manage.py runserver

```


```bash
npx create-react-app my-labeling-app
cd my-labeling-app
npm install react react-dom
git clone https://github.com/HumanSignal/label-studio.git
cp -r label-studio/web/libs/editor src/label-studio-editor
cd src/label-studio-editor
npm install
cd../../
# create src/App.js
npm start
#When you're ready to deploy your application, you'll need to create a production build using npm run build.
```

Import and use the editor component:
```bash
// src/App.js
import React from 'react';
import LabelStudio from './label-studio-editor';

function App() {
  return (
    <div>
      <LabelStudio />
    </div>
  );
}

export default App;
```

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