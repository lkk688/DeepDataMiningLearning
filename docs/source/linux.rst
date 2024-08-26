Linux Machine Setup
====================

https://docs.google.com/document/d/17vIhJuqDILWZhxh3WPh_voTGPPe-221xAEj_djWgSJE/edit

Get System information
---------------------

.. code-block:: console

    lkk@p100:~$ uname -m && cat /etc/*release
    x86_64
    DISTRIB_ID=Ubuntu
    DISTRIB_RELEASE=22.04
    DISTRIB_CODENAME=jammy
    DISTRIB_DESCRIPTION="Ubuntu 22.04.4 LTS"
    PRETTY_NAME="Ubuntu 22.04.4 LTS"
    NAME="Ubuntu"
    VERSION_ID="22.04"
    VERSION="22.04.4 LTS (Jammy Jellyfish)"
    VERSION_CODENAME=jammy
    ID=ubuntu
    ID_LIKE=debian
    HOME_URL="https://www.ubuntu.com/"
    SUPPORT_URL="https://help.ubuntu.com/"
    BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
    PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
    UBUNTU_CODENAME=jammy
    $ systemctl suspend #put system to sleep

Install Python
-----------------
Python 3.8-3.11 is generally installed by default on any of our supported Linux distributions, if you want to install another version, there are multiple ways: 1) APT: `sudo apt install python` or 2) Download from [Python](https://www.python.org/downloads/)

Python two supported package managers: Anaconda or pip. Anaconda is the recommended package manager as it will provide you all of the PyTorch dependencies in one, sandboxed install, including Python. While Python 3.x is installed by default on Linux, pip is not installed by default: `sudo apt install python3-pip`. 

.. code-block:: console

    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    sh Miniconda3-latest-Linux-x86_64.sh

Install NVIDIA driver
---------------------

.. code-block:: console

    $ sudo ubuntu-drivers devices
    $ sudo apt install nvidia-driver-545 #for desktop
    #for server: sudo apt install nvidia-driver-550-server

If your nvidia-smi show "Failed to initialize NVML: Driver/library version mismatch", you need to install a matched nvidia driver version.

Install Basic Software
-----------------------

.. code-block:: console

    $ sudo apt install openssh-server
    $ sudo apt install net-tools
    $ sudo dpkg -i google-chrome-stable_current_amd64.deb
    $ sudo dpkg -i code_1.74.3-1673284829_amd64.deb
    $ sudo apt install -y ./teamviewer_15.37.3_amd64.deb
    $ sudo apt-get install libgtkglext1
    $ sudo dpkg -i anydesk_6.2.1-1_amd64.deb 
    $ sudo apt install ffmpeg
    $ sudo apt install curl

Networking
-----------
Check current IP address via `ifconfig` or `ip route show | grep -i default | awk '{ print $3}'`

To view all TCP or UDP ports that are being listened on, along with the associated services and socket status: `sudo netstat -tunlp`

Download Cisco Anyconnect VPN client from SJSU: https://vpn.sjsu.edu/CACHE/stc/1/index.html

.. code-block:: console

    lkk@lkk-intel12:~/Downloads$ sudo bash anyconnect-linux64-4.10.03104-core-vpn-webdeploy-k9.sh

If Anyconnect cannot open browser, ref: https://community.cisco.com/t5/vpn/cisco-anyconnect-mobility-client-linux-not-opening-browser/td-p/4783285

.. code-block:: console

    sudo apt-get purge firefox
    cd ;
    sudo rm -rf .mozilla
    cd /etc
    sudo rm -rf firefox
    cd /usr/lib/
    sudo rm -rf firefox-addons

    sudo apt install firefox

NVIDIA CUDA
------------
Install NVIDIA CUDA , ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

To verify that your GPU is CUDA-capable, go to your distribution's equivalent of System Properties, or, from the command line, enter:

.. code-block:: console

    $ lspci | grep -i nvidia
    $ uname -m && cat /etc/*release
    $ gcc --version #To verify the version of gcc installed on your system
    $ uname -r #The version of the kernel your system is running

There are multiple ways to install cuda, here we use Conda to install the specific version (11.8) of the cuda: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#conda-installation

.. code-block:: console

    (base) lkk@lkk-intel13:~$ conda activate mycondapy310
    $ conda install cuda -c nvidia/label/cuda-11.8.0
    (mycondapy310) lkk@lkk-intel13:~$ nvcc -V
    $ nvcc -V                                                                
    nvcc: NVIDIA (R) Cuda compiler driver                                                                    
    Copyright (c) 2005-2022 NVIDIA Corporation                                                               
    Built on Wed_Sep_21_10:33:58_PDT_2022                                                                    
    Cuda compilation tools, release 11.8, V11.8.89                                                           
    Build cuda_11.8.r11.8/compiler.31833905_0 

Install CUDNN (only required for Tensorflow). Check [TensorflowGPU](https://www.tensorflow.org/install/source#tested_build_configurations) for required CUDNN versions (e.g., tensorflow-2.14.0 support cudnn8.7 and cuda11.8). Install latest [cudnn](https://docs.nvidia.com/deeplearning/cudnn/latest/) or [cudnn8.7](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-870/index.html).

If only install cudnn python runtime, you can install from [nvidia-cudnn-cu11](https://pypi.org/project/nvidia-cudnn-cu11/#history), e.g., `pip install nvidia-cudnn-cu11==8.7.0.84`

NVIDIA Docker
--------------

NVIDIA docker installation: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

.. code-block:: console

    sudo snap install curl
    curl https://get.docker.com | sh \
    && sudo systemctl --now enable docker

    #https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user
	sudo usermod -aG docker $USERv
	Log out and log back in so that your group membership is re-evaluated.
	lkk@lkk-intel12:~$ newgrp docker
	lkk@lkk-intel12:~$ docker run hello-world

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
        && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
            
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker

    sudo docker run --rm --gpus all nvidia/cuda:11.7.1-devel-ubuntu22.04 nvidia-smi

    docker pull nvidia/cuda:11.7.1-devel-ubuntu22.04

System Upgrade

-----------------------
.. code-block:: console
    sudo apt update
    sudo apt upgrade
    $ sudo apt dist-upgrade
    $ do-release-upgrade

    To make recovery in case of failure easier, an additional sshd will
    be started on port '1022'. If anything goes wrong with the running
    ssh you can still connect to the additional one.
    If you run a firewall, you may need to temporarily open this port. As
    this is potentially dangerous it's not done automatically. You can
    open the port with e.g.:
    'iptables -I INPUT -p tcp --dport 1022 -j ACCEPT'

Pytorch, Tensorflow, and Huggingface installation
------------------------
Install pytorch: https://pytorch.org/get-started/locally/

Check the supported CUDA, CUDNN version for a specific Tensorflow version: [Tensorflow Versions](https://www.tensorflow.org/install/source#tested_build_configurations)

.. code-block:: console

    $ conda -V # check version
    $ conda info --envs #Check available conda environments
    (base) lkk@lkk-intel12:~$ conda create -n py311cu124 python=3.11
    (base) lkk@lkk-intel12:~$ conda activate py311cu124
    (py311cu124) lkk@lkk-intel12:~$ conda install -y cuda -c nvidia/label/cuda-12.4
    (py311cu124) lkk@lkk-intel12:~$ nvcc -V
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2024 NVIDIA Corporation
    Built on Thu_Mar_28_02:18:24_PDT_2024
    Cuda compilation tools, release 12.4, V12.4.131
    Build cuda_12.4.r12.4/compiler.34097967_0
    (py311cu124) lkk@lkk-intel12:~$ conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

    (py311cu124) lkk@lkk-intel12:~$ pip install tensorflow[and-cuda]
    # Verify Pytorch installation
    python3 -c "import torch; print('Torch version:', torch.__version__); print(torch.cuda.is_available())"

    # Verify Tensorflow installation
    python3 -c "import tensorflow as tf; print('tf version:', tf.__version__); print(tf.config.list_physical_devices('GPU'))"

    (py311cu124) lkk@lkk-intel12:~$ conda install -y cuda -c nvidia/label/cuda-12.2 #new method from https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#conda-installation
    
Old version:
.. code-block:: console

    $ conda activate mycondapy310
    $ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    #verify pytorch installation
    $ python
    >>> import torch                                                                                         
    >>> print(torch.__version__)
    2.3.0                                                                                                    
    >>> torch.cuda.is_available()                                                                            
    True

Install Tensorflow:

.. code-block:: console

    python3 -m pip install tensorflow[and-cuda]
    #pip install tensorflow[and-cuda]==2.14.0

Install TensorRT: https://developer.nvidia.com/tensorrt, using the tar methold installation: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar, select "TensorRT 10.3 GA for Linux x86_64 and CUDA 12.0 to 12.5 TAR Package"

.. code-block:: console

    (py311cu124) lkk@lkk-intel12:~/Developer$ tar -xzvf TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.5.tar.gz
    (py311cu124) lkk@lkk-intel12:~/Developer$ ls TensorRT-10.3.0.26
    bin  data  doc  include  lib  python  samples  targets
    (py311cu124) lkk@lkk-intel12:~/Developer$ export LD_LIBRARY_PATH=/home/lkk/Developer/TensorRT-10.3.0.26/lib:$LD_LIBRARY_PATH
    (py311cu124) lkk@lkk-intel12:~/Developer/TensorRT-10.3.0.26/python$ python3 -m pip install tensorrt-10.3.0-cp311-none-linux_x86_64.whl 
    Processing ./tensorrt-10.3.0-cp311-none-linux_x86_64.whl
    Installing collected packages: tensorrt
    Successfully installed tensorrt-10.3.0
    (py311cu124) lkk@lkk-intel12:~/Developer/TensorRT-10.3.0.26/python$ python3 -m pip install numpy onnx onnx-graphsurgeon
    #$ strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
    (py311cu124) lkk@lkk-intel12:~/Developer/TensorRT-10.3.0.26/samples/build$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lkk/miniconda3/lib #solve the glibc problem
    $ conda config --set channel_priority flexible 
    $ conda install -c anaconda scipy
    (py311cu124) lkk@lkk-intel12:~/Developer/TensorRT-10.3.0.26/samples/build$ strings ~/miniconda3/lib/libstdc++.so.6 | grep GLIBCXX_3.4.30
    (py311cu124) lkk@lkk-intel12:~/Developer/TensorRT-10.3.0.26/samples/build$ strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4.30
    GLIBCXX_3.4.30

    (py311cu124) lkk@lkk-intel12:~/Developer/TensorRT-10.3.0.26/samples/build$ cmake .. -DTRT_OUT_DIR=`pwd`/out
    cmake: /home/lkk/miniconda3/lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by cmake)



setup jupyter:

.. code-block:: console

    $ conda install -c conda-forge scikit-learn tensorboard jupyterlab
    $ conda install ipykernel
    ipython kernel install --user --name=py311cu124
    $ jupyter kernelspec list #view current jupyter kernels

Install huggingface transformers: https://huggingface.co/docs/accelerate/basic_tutorials/install

.. code-block:: console

    $ pip install transformers #conda install -c huggingface transformers
    #pip install -U transformers --upgrade
    $ pip install accelerate # conda install -c conda-forge accelerate
    $ pip install evaluate
    #$ pip install cchardet
    $ conda install -c conda-forge umap-learn #pip install umap-learn
    $ pip install portalocker
    $ pip install torchdata
    $ pip install torchtext
    $ pip install configargparse
    $ pip install datasets # conda install -c huggingface -c conda-forge datasets
    $ pip install torchinfo
    $ pip install gputil

    pip install pyyaml seaborn scikit-image onnx onnx-simplifier onnxruntime
    pip install sacrebleu sacremoses nltk rouge_score
    pip install soundfile #for audio
    pip install librosa #changed numpy-1.26.2 to 1.24.4
    pip install jiwer #evaluate using the word error rate (WER) metric

    conda install chardet #solve problem of "ImportError: cannot import name 'get_full_repo_name' from 'huggingface_hub'"
    pip install safetensors
    conda install -c conda-forge tokenizers

   % accelerate config
      Do you wish to use FP16 or BF16 (mixed precision)?                                                                                                          
   bf16                                                                                                                                                        
   accelerate configuration saved at /Users/kaikailiu/.cache/huggingface/accelerate/default_config.yaml 
   % accelerate env
   
   $ conda install -c conda-forge spacy #https://spacy.io/usage
   #$ conda install -c conda-forge cupy #https://docs.cupy.dev/en/stable/install.html
   $ python -m spacy download en_core_web_sm
   >>> import spacy
   >>> spacy.prefer_gpu()
   True
   >>> nlp = spacy.load("en_core_web_sm")
   
   
Install Ollama: https://github.com/ollama/ollama/blob/main/docs/linux.md

.. code-block:: console

    (mycondapy310) lkk@lkk-intel12:~/Developer$ curl -fsSL https://ollama.com/install.sh | sh
    >>> Downloading ollama...
    >>> Installing ollama to /usr/local/bin...
    [sudo] password for lkk: 
    >>> Creating ollama user...
    >>> Adding ollama user to render group...
    >>> Adding ollama user to video group...
    >>> Adding current user to ollama group...
    >>> Creating ollama systemd service...
    >>> Enabling and starting ollama service...
    Created symlink /etc/systemd/system/default.target.wants/ollama.service → /etc/systemd/system/ollama.service.
    >>> NVIDIA GPU installed.
    (mycondapy310) lkk@lkk-intel12:~/Developer$ ollama run llama3



   

Disk
----------

Check disk space:

.. code-block:: console

    $ df -H
    Filesystem      Size  Used Avail Use% Mounted on
    tmpfs            14G  3.4M   14G   1% /run
    /dev/nvme0n1p2  2.0T   83G  1.8T   5% /
    tmpfs            68G  549k   68G   1% /dev/shm
    tmpfs           5.3M  4.1k  5.3M   1% /run/lock
    /dev/nvme0n1p1  536M  6.4M  530M   2% /boot/efi
    tmpfs            14G  156k   14G   1% /run/user/1000
    tmpfs            14G  140k   14G   1% /run/user/1001
    $ lsblk
    NAME        MAJ:MIN RM   SIZE RO TYPE MOUNTPOINTS
    loop0         7:0    0     4K  1 loop /snap/bare/5
    loop1         7:1    0 244.5M  1 loop /snap/firefox/2800
    loop2         7:2    0  63.5M  1 loop /snap/core20/1891
    loop4         7:4    0  73.8M  1 loop /snap/core22/750
    loop5         7:5    0  73.9M  1 loop /snap/core22/766
    loop6         7:6    0 244.8M  1 loop /snap/firefox/2760
    loop7         7:7    0 349.7M  1 loop /snap/gnome-3-38-2004/137
    loop8         7:8    0 349.7M  1 loop /snap/gnome-3-38-2004/140
    loop9         7:9    0  53.3M  1 loop /snap/snapd/19457
    loop10        7:10   0 460.7M  1 loop /snap/gnome-42-2204/105
    loop11        7:11   0  91.7M  1 loop /snap/gtk-common-themes/1535
    loop12        7:12   0  45.9M  1 loop /snap/snap-store/638
    loop13        7:13   0  12.3M  1 loop /snap/snap-store/959
    loop14        7:14   0  63.4M  1 loop /snap/core20/1950
    loop15        7:15   0  53.3M  1 loop /snap/snapd/19361
    loop16        7:16   0   428K  1 loop /snap/snapd-desktop-integration/57
    loop17        7:17   0   452K  1 loop /snap/snapd-desktop-integration/83
    loop18        7:18   0 466.5M  1 loop /snap/gnome-42-2204/111
    sda           8:0    0   9.1T  0 disk 
    nvme0n1     259:0    0   1.8T  0 disk 
    ├─nvme0n1p1 259:1    0   512M  0 part /boot/efi
    └─nvme0n1p2 259:2    0   1.8T  0 part /var/snap/firefox/common/host-hunspell

Mount a new disk. You can see disk "sda" from the "lsblk" is not mounted. 

.. code-block:: console

    lkk@lkk-intel13:/$ sudo mkdir DATA10T

    lkk@lkk-intel13:/$ sudo nano -Bw /etc/fstab
    /dev/sda        /DATA10T        ext4    defaults        0       2
    lkk@lkk-intel13:/$ sudo mount -a
    (base) lkk@lkk-intel13:/$ df -H
    Filesystem      Size  Used Avail Use% Mounted on
    tmpfs            14G  3.4M   14G   1% /run
    /dev/nvme0n1p2  2.0T   83G  1.8T   5% /
    tmpfs            68G  549k   68G   1% /dev/shm
    tmpfs           5.3M  4.1k  5.3M   1% /run/lock
    /dev/nvme0n1p1  536M  6.4M  530M   2% /boot/efi
    tmpfs            14G  156k   14G   1% /run/user/1000
    tmpfs            14G  140k   14G   1% /run/user/1001
    /dev/sda         10T   37k  9.5T   1% /DATA10T

Check directory size:

.. code-block:: console

    du -sh /path/to/directory

sshfs
-------

.. code-block:: console

    lkk@lkk-intel12:~/Documents/Dataset/Kitti$ sudo apt-get install sshfs
    lkk@lkk-intel12:~/Documents/Dataset/HPC249Data$ sshfs 010796032@coe-hpc2.sjsu.edu:/data/cmpe249-fa22 .


Network
-------

You can use the following Linux commands to scan for IP addresses of other machines in your local network:

.. code-block:: console

    #option1:
    sudo apt-get install arp-scan
    sudo arp-scan --interface=eth0 --localnet
    #option2:
    sudo apt-get install nmap
    sudo nmap -sn 130.65.157.0/24

Network Tunnel
--------------

Access a colab instance in a remote machine

.. code-block:: console

    $ ssh lkk@lkk-intel12 #ssh into that remote machine
    # run a colab docker image
    $ docker run --gpus=all -p 127.0.0.1:9000:8080 us-docker.pkg.dev/colab-images/public/runtime
    # check the output jupyter link with token. 
    # Create a new terminal in your local machine, create ssh tunnel to the remote machine
    ssh -L 9000:localhost:9000 lkk@lkk-intel12
    # Open Colab in your local machine, click connect to a local instance via this link:
    http://localhost:9000/?token=829d73b43e0954bbf277956aeca4964494c04d6ef7f58016

Install zrok (https://docs.zrok.io/docs/guides/install/linux/):

.. code-block:: console

    $ sudo apt  install curl
    $ (set -euo pipefail;

    curl -sSLf https://get.openziti.io/tun/package-repos.gpg \
    | sudo gpg --dearmor --output /usr/share/keyrings/openziti.gpg;
    sudo chmod a+r /usr/share/keyrings/openziti.gpg;

    sudo tee /etc/apt/sources.list.d/openziti-release.list >/dev/null <<EOF;
    deb [signed-by=/usr/share/keyrings/openziti.gpg] https://packages.openziti.org/zitipax-openziti-deb-stable debian main
    EOF

    sudo apt update;
    sudo apt install zrok;
    zrok version;
    )
    $ zrok invite

After you enter your email address, you will get the registration email from zrok and create a new account. When your zrok account was created, the service generated a secret token that identifies and authenticates in a single step. You can also get this command from the web console, click on your email address in the upper right corner of the header. That drop down menu contains an Enable Your Environment link.

.. code-block:: console

    To enable your shell for zrok, use this command:
    $ zrok enable XXXX 
    ⡿  the zrok environment was successfully enabled...
    $ zrok status

If we return to the web console, we'll now see the new environment reflected in the explorer view.

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

Docker
------

Install `Docker <https://docs.docker.com/engine/install/ubuntu/>`, follow `Post-installation steps for Linux <https://docs.docker.com/engine/install/linux-postinstall/>`

Install NVIDIA container toolkit `container toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt>`
Use the automatic script to install docker:

.. code-block:: console

    curl -fsSL https://get.docker.com -o install-docker.sh
    sudo sh install-docker.sh
    sudo docker run hello-world #test docker
    $ sudo groupadd docker
    $ sudo usermod -aG docker $USER
    $ newgrp docker #activate changes
    #chmod +x docker_post.sh
    #sudo ./docker_post.sh
    docker run hello-world #test docker without sudo
    #exit the docker via Ctrl+D or exit

Install Nvidia container toolkit via APT:

.. code-block:: console

    $ curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    $ sudo apt-get update
    $ sudo apt-get install -y nvidia-container-toolkit
    #Configure the container runtime by using the nvidia-ctk command:
    $ sudo nvidia-ctk runtime configure --runtime=docker
    $ sudo systemctl restart docker
    #To configure the container runtime for Docker running in Rootless mode
    #$ nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json #this may cause the NVML unknown error
    $ sudo systemctl restart docker
    #Configure /etc/nvidia-container-runtime/config.toml by using the sudo nvidia-ctk command:
    $ sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place
    #Configuring containerd (for Kubernetes) -- optional
    $ sudo nvidia-ctk runtime configure --runtime=containerd
    $ sudo systemctl restart docker
    #Run a sample CUDA container:
    $ docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
    #if you see the error of "Failed to initialize NVML: Unknown Error"
    $ sudo nano /etc/nvidia-container-runtime/config.toml #set the parameter no-cgroups = false


We also put the post installation of Docker and install nvidia container in one script file:

.. code-block:: console

    ./install_postnvidiacontainer.sh #perform post installation of docker and install nvidia-container toolkit
    #Wrote updated config to /etc/docker/daemon.json

Run a sample CUDA container:

.. code-block:: console

    systemctl restart docker #optional
    docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi

Clean up any resources (images, containers, volumes, and networks) that are dangling

.. code-block:: console

    docker system prune
    #To additionally remove any stopped containers and all unused images (not just dangling images), add the -a flag:
    docker system prune -a
    #Using docker rmi (if you want to remove specific images): docker rmi ImageID_or_Tag

Run Google Colab container image:
.. code-block:: console
    
    $ docker run -p 127.0.0.1:9000:8080 --rm --runtime=nvidia --gpus all us-docker.pkg.dev/colab-images/public/runtime

In another terminal, you can check the new container via "docker images", note down the image id, and run this image: `docker run -it --rm 486a56765aad`. For Colab image, you can connect to the colab container runtime in Colab and make changes.

After you entered the container and did changes inside the container, click "control+P+Q" to exit the container without terminate the container. Use "docker ps" to check the container id, then use "docker commit" to commit changes:

.. code-block:: console
    
    docker commit -a "Kaikai Liu" -m "Mymodified colab container" a43c343f7f7c mycolab:v1

Now, you can see your newly created container image named "mycolab:v1" in "docker images".

After you modified the container, you can use the command "docker exec -it container_id /bin/bash" to get a bash shell in the container. We also created a shell script to enter into the colab container:

.. code-block:: console

    DeepDataMiningLearning/docker$ ./runcolabcontainer.sh mycolab:v1

Popular Docker commands:
 * Stop a running container: docker stop container_id
 * Stop all containers not running: docker container prune
 * Delete docker images: docker image rm dockerimageid

Add New Sudo Users
------------------
Ref: https://www.digitalocean.com/community/tutorials/how-to-create-a-new-sudo-enabled-user-on-ubuntu-22-04-quickstart

.. code-block:: console

    lkk@p100:~$ sudo adduser student
    [sudo] password for lkk: 
    Adding user `student' ...
    Adding new group `student' (1001) ...
    Adding new user `student' (1001) with group `student' ...
    Creating home directory `/home/student' ...
    Copying files from `/etc/skel' ...
    New password: 
    #add to sudo group
    lkk@p100:~$ sudo usermod -aG sudo student
    #To test that the new sudo permissions are working, first use the su command to switch to the new user account:
    lkk@p100:~$ su - student

Git
----

To commit changes for a single file using Git

.. code-block:: console
    
    git config --global user.email "kaikai.liu@sjsu.edu"
    git config --global user.name "Kaikai Liu"
    git add path/to/your/file.ext
    git commit -m "Your commit message here"
    git push #enter user name and password (use token instead of the actual password) of Github


.. code-block:: console

    git checkout -- <file> #To discard changes in a specific file using Git
    #If you want to discard changes in all files in the working directory, you can run:
    git restore .
    git pull #get the updates


NodeJS Install
--------------

.. code-block:: console

    $ sudo apt install nodejs
    $ sudo apt install npm
    #install localtunnel: https://www.npmjs.com/package/localtunnel
    npm install localtunnel
    $ npx localtunnel --port 8501
    your url is: https://neat-taxis-fetch.loca.lt

Sync Folder
------------
rsync command: https://www.redhat.com/sysadmin/sync-rsync

.. code-block:: console

    $ rsync -rtvuP ImageClassData/ ../rnd-liu/Datasets/ImageClassData/

VSCode Extensions:

"Black Formatter" to automatic code

Common errors
-------------

.. code-block:: console

    (mycondapy310) lkk@lkk-intel13:~/Developer/3DDepth$ python ./VisUtils/testmayavi.py
    qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/lkk/miniconda3/envs/mycondapy310/lib/python3.10/site-packages/cv2/qt/plugins" even though it was found.
    This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

    Available platform plugins are: xcb, eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx, webgl

    pip3 install --upgrade pyside2 pyqt5
    pip uninstall opencv-python
    pip uninstall opencv-python-headless
    pip install opencv-python-headless

    python ./VisUtils/testmayavi.py
    libGL error: No matching fbConfigs or visuals found
    libGL error: failed to load driver: swrast

Fix nvidia-smi error of "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver", reinstall the nvidia driver

.. code-block:: console

    sudo ubuntu-drivers devices #check the latest and recommended driver version
    sudo apt install nvidia-driver-545
    sudo reboot