Container
==========

.. _container:

Author:
   * *Kaikai Liu*, Associate Professor, SJSU
   * **Email**: kaikai.liu@sjsu.edu
   * **Web**: http://www.sjsu.edu/cmpe/faculty/tenure-line/kaikai-liu.php


Container Installation
----------------------
Install Docker: https://docs.docker.com/engine/install/ubuntu/ and follow Post-installation steps for Linux: https://docs.docker.com/engine/install/linux-postinstall/

Setup Docker and nvidia container runtime via nvidiacontainer: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html, https://docs.nvidia.com/dgx/nvidia-container-runtime-upgrade/index.html

NVIDIA docker installation: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

.. code-block:: console

    sudo snap install curl
    curl https://get.docker.com | sh \
    && sudo systemctl --now enable docker

    #https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user
	sudo usermod -aG docker $USER
	Log out and log back in so that your group membership is re-evaluated.
	lkk@lkk-intel12:~$ newgrp docker
	lkk@lkk-intel12:~$ docker run hello-world

    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker

    sudo docker run --rm --gpus all nvidia/cuda:11.7.1-devel-ubuntu22.04 nvidia-smi
    
    docker pull nvidia/cuda:11.7.1-devel-ubuntu22.04
    docker pull nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
    docker run --gpus all -it nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
    docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 nvidia-smi
    #https://hub.docker.com/r/nvidia/cuda
    docker run --rm --gpus all nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 nvidia-smi
    #run a google colab container
     docker run --runtime=nvidia --gpus all -p 127.0.0.1:9000:8080 us-docker.pkg.dev/colab-images/public/runtime


After you build the container, you can check the new container via "docker images", note down the image id, and run this image:

.. code-block:: console

    sudo docker run -it --rm 486a56765aad

After you entered the container and did changes inside the container, click "control+P+Q" to exit the container without terminate the container. Use "docker ps" to check the container id, then use "docker commit" to commit changes:

.. code-block:: console

    docker commit -a "Kaikai Liu" -m "First ROS2-x86 container" 196073a381b4 myros2:v1

Now, you can see your newly created container image named "myros2:v1" in "docker images".

You can now start your ROS2 container (i.e., myros2:v1) via runcontainer.sh, change the script file if you want to change the path of mounted folders. 
.. code-block:: console

    sudo xhost +si:localuser:root
    ./scripts/runcontainer.sh [containername]

after you 
Re-enter a container: use the command "docker exec -it container_id /bin/bash" to get a bash shell in the container.

Popular Docker commands:
 * Stop a running container: docker stop container_id
 * Stop all containers not running: docker container prune
 * Delete docker images: docker image rm dockerimageid

To delete all containers including its volumes use

.. code-block:: console

    docker rm -vf $(docker ps -aq)

To delete all the images

.. code-block:: console

    docker rmi -f $(docker images -aq)
    docker images -a -q | % { docker image rm $_ -f } #Windows - Powershell


Use the Dockerfile under scripts folder to build the container image:

.. code-block:: console

    myROS2/docker$ docker build -t myros2ubuntu22cuda117 .

You can also use prune to delete everything

.. code-block:: console

    docker system prune -a --volumes

You can also build the docker image via docker vscode extension. After the extension is installed, simply right click the Dockerfile and select "build image"
