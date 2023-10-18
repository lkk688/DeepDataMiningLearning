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
