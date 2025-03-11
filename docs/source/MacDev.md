# Development Environment for Apple Silicon

## Environment
```bash
(mypy310) kaikailiu@kaikais-mbp ~ % nano ~/.zshrc
(mypy310) kaikailiu@kaikais-mbp ~ % source ~/.zshrc
```

## Xcode Tools
Install Xcode command-line tools (CLT), this includes all the useful open-source tools Unix developers expect, like git, clang, and more.
```bash
xcode-select --install
```

## Brew
Install [Brew](https://brew.sh/): package manager for Macs
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
The script will install Homebrew to the correct directory for Apple silicon, which is /opt/homebrew

After the installation is complete, the terminal will give you next steps. The terminal will likely give you a line to add to your .zshrc file. To do this, open your .zshrc file in a text editor. 
```bash
echo >> /Users/kaikailiu/.zprofile
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/kaikailiu/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
brew help #test brew
```

Install LLVM:
```bash
brew install llvm
brew --prefix llvm
```
If you need to have llvm first in your PATH, run:
  echo 'export PATH="/opt/homebrew/opt/llvm/bin:$PATH"' >> ~/.zshrc

For compilers to find llvm you may need to set:
  export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
  export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
  
Then append /lib/libLLVM.dylib to the output of that command.
For example, if brew --prefix llvm outputs /usr/local/opt/llvm, then the full path would be /usr/local/opt/llvm/lib/libLLVM.dylib.

Set the DRJIT_LIBLLVM_PATH Environment Variable:
```bash
export DRJIT_LIBLLVM_PATH=/path/to/libLLVM.dylib
```
Open the file in a text editor (e.g., nano ~/.zshrc).
Add the line: export DRJIT_LIBLLVM_PATH=/path/to/libLLVM.dylib

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
% conda env list
% conda env remove --name myenv
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

### LLM related Packages
```bash
pip install langchain
pip install -qU langchain-openai
pip install -qU langchain-google-vertexai
pip install -qU langchain-nvidia-ai-endpoints
pip install langchain-pinecone
pip install langchain_community
pip install langchain-chroma
pip install openai
pip install streamlit
pip install tiktoken
pip install pypdf
```
### Google Cloud
Install the Google Cloud CLI: https://cloud.google.com/sdk/docs/install-sdk
```bash
curl https://sdk.cloud.google.com | bash
% ls ~/google-cloud-sdk
google-cloud-sdk % ./bin/gcloud init
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
$ zrok share private --backend-mode tcpTunnel 192.168.137.14:22
#it will show access your share with: zrok access private gdjf3oz1pudh
```

In your local Mac, enter `zrok access private gdjf3oz1pudh`, it will show `tcp://127.0.0.1:9191 -> gdjf3oz1pudh`. In another Mac terminal, ssh into the remote machine
```bash
% ssh lkk@127.0.0.1 -p 9191
# same to this: ssh lkk@localhost -p 9191
```

## Problems
* "torchvision.ops box_ops Couldn't load custom C++ ops.": upgrade torch and torchvision to the same version.

## Ollama
Download Ollama for Mac, Once the download is complete, locate the .zip file and extract its contents. This should create Ollama.app. Drag Ollama.app to your Applications folder. Open the Applications folder and double-click on Ollama.app. Follow the setup wizard to complete the installation. The wizard will prompt you to install the command line version (ollama). Verify Olllama and make sure its working
```bash
  % ollama --version
  ollama version is 0.5.7
  % ollama run llama3.2
  >>> /bye
  % ollama show llama3.2
  % ollama list #List models on your computer
  % ollama ps
  % ollama stop llama3.2
```

Try accessing it as well : http://127.0.0.1:11434/

https://github.com/ollama/ollama

https://ollama.com/library/deepseek-r1

Install OpenWebui: https://github.com/open-webui/open-webui
```bash
% conda create --name mypy311 python=3.11
% conda activate mypy311
pip install open-webui
pip install ollama
% open-webui serve                   
Loading WEBUI_SECRET_KEY from file, not provided as an environment variable.
Generating a new secret key and saving it to /Users/kaikailiu/.webui_secret_key
Loading WEBUI_SECRET_KEY from /Users/kaikailiu/.webui_secret_key
```

To manage your Ollama instance in Open WebUI, Go to Admin Settings in Open WebUI. Navigate to Connections > Ollama > Manage (click the wrench icon).

```bash
ollama run deepseek-r1:32b --verbose
(base) kaikailiu@Kaikais-MacBook-Pro ~ % ollama ps                 
NAME               ID              SIZE     PROCESSOR    UNTIL              
deepseek-r1:32b    38056bbcbb2d    21 GB    100% GPU     4 minutes from now
```

On Mac, the models will be download to ~/.ollama/models; On Linux (or WSL), the models will be stored at /usr/share/ollama/.ollama/models

Ollama with Langchain: https://python.langchain.com/docs/integrations/chat/ollama/