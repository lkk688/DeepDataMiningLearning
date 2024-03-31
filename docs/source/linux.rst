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
.. code-block:: console

    (base) lkk@lkk-intel13:~$ conda activate mycondapy310
    (mycondapy310) lkk@lkk-intel13:~$ nvcc -V
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2022 NVIDIA Corporation
    Built on Wed_Sep_21_10:33:58_PDT_2022
    Cuda compilation tools, release 11.8, V11.8.89
    Build cuda_11.8.r11.8/compiler.31833905_0

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
======
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

Docker
------

Install `Docker <https://docs.docker.com/engine/install/ubuntu/>`, follow `Post-installation steps for Linux <https://docs.docker.com/engine/install/linux-postinstall/>`

Install NVIDIA container toolkit `container toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt>`
Use the automatic script to install docker:

.. code-block:: console

    curl -fsSL https://get.docker.com -o install-docker.sh
    sudo sh install-docker.sh
    sudo docker run hello-world #test docker
    chmod +x docker_post.sh
    sudo ./docker_post.sh
    docker run hello-world #test docker without sudo
    #exit the docker via Ctrl+D or exit
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