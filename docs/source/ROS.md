
# NVIDIA ISAAC ROS
Isaac ROS Dev is a container-based workflow. The Isaac ROS development environment is run within a Docker container on the target platform using the script run_dev.sh. After you’ve setup your ROS workspace on your machine with your packages, third party packages, and one or many Isaac ROS packages, you can use this script to launch a Docker container that contains all the system libraries and dependencies to build.

ref: https://nvidia-isaac-ros.github.io/getting_started/index.html, https://nvidia-isaac-ros.github.io/concepts/docker_devenv/index.html#development-environment
```bash
sudo systemctl daemon-reload && sudo systemctl restart docker
sudo apt-get install git-lfs
git lfs install --skip-repo
#Create ISAAC_ROS_WS environmental variable
mkdir -p ~/workspaces/isaac_ros-dev/src
echo "export ISAAC_ROS_WS=${HOME}/workspaces/isaac_ros-dev/" >> ~/.bashrc
source ~/.bashrc
#get ISAAC ROS common repo
~/workspaces/isaac_ros-dev/src$ git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
~/workspaces/isaac_ros-dev/src/isaac_ros_common$ scripts/run_dev.sh -d ~/workspaces/isaac_ros-dev
Building /home/lkk/workspaces/isaac_ros-dev/src/isaac_ros_common/scripts/../docker/Dockerfile.user as image: isaac_ros_dev-x86_64 with base: ros2_humble-image
[+] Building 22.4s (16/16) FINISHED
admin@lkk-intel12:/workspaces/isaac_ros-dev$ #inside the docker
admin@lkk-intel12:/workspaces/isaac_ros-dev$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Aug_15_22:02:13_PDT_2023
Cuda compilation tools, release 12.2, V12.2.140
Build cuda_12.2.r12.2/compiler.33191640_0
admin@lkk-intel12:/workspaces/isaac_ros-dev$ pip list | grep torch
torch                                2.4.0+cu121
torchaudio                           2.4.0+cu121
torchvision                          0.19.0+cu121
admin@lkk-intel12:/workspaces/isaac_ros-dev$ ros2 topic list
/parameter_events
/rosout
admin@lkk-intel12:/workspaces/isaac_ros-dev$ echo $ISAAC_ROS_WS
/workspaces/isaac_ros-dev
```

[run_dev.sh](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/scripts/run_dev.sh) sets up a development environment containing ROS 2 and key versions of NVIDIA frameworks prepared for both x86_64 and Jetson. This script prepares a Docker image with a supported configuration for the host machine and delivers you into a bash prompt running inside the container. From here, you are ready to execute ROS 2 build/run commands with your host workspace files, which are mounted into the container and available to edit both on the host and in the container.

Download quickstart data from NGC (ref: https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_apriltag/isaac_ros_apriltag/index.html#quickstart):
```bash
admin@lkk-intel12:/workspaces/isaac_ros-dev$ sudo apt-get install -y curl tar
NGC_ORG="nvidia"
NGC_TEAM="isaac"
NGC_RESOURCE="isaac_ros_assets"
NGC_VERSION="isaac_ros_apriltag"
NGC_FILENAME="quickstart.tar.gz"

REQ_URL="https://api.ngc.nvidia.com/v2/resources/$NGC_ORG/$NGC_TEAM/$NGC_RESOURCE/versions/$NGC_VERSION/files/$NGC_FILENAME"

mkdir -p ${ISAAC_ROS_WS}/isaac_ros_assets/${NGC_VERSION} && \
    curl -LO --request GET "${REQ_URL}" && \
    tar -xf ${NGC_FILENAME} -C ${ISAAC_ROS_WS}/isaac_ros_assets/${NGC_VERSION} && \
    rm ${NGC_FILENAME}
```

## Run the Apriltag example 
Ref [apriltag](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag): Build isaac_ros_apriltag, make sure the Docker container is already launched using the run_dev.sh script: `cd ${ISAAC_ROS_WS}/src/isaac_ros_common && ./scripts/run_dev.sh`
```bash
cd ${ISAAC_ROS_WS}/src && \
   git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag.git
admin@lkk-intel12:/workspaces/isaac_ros-dev/src$ rosdep install --from-paths ${ISAAC_ROS_WS}/src/isaac_ros_apriltag --ignore-src -y
#build the package from source
admin@lkk-intel12:/workspaces/isaac_ros-dev/src$ cd ${ISAAC_ROS_WS}/ && \
   colcon build --symlink-install --packages-up-to isaac_ros_apriltag
#Source the ROS workspace:
admin@lkk-intel12:/workspaces/isaac_ros-dev$ source install/setup.bash
admin@lkk-intel12:/workspaces/isaac_ros-dev$ sudo apt-get install -y ros-humble-isaac-ros-examples
admin@lkk-intel12:/workspaces/isaac_ros-dev$ ros2 launch isaac_ros_examples isaac_ros_examples.launch.py launch_fragments:=apriltag interface_specs_file:=${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_apriltag/quickstart_interface_specs.json
```

Play the ROS bag Open a second terminal inside the Docker container:
```bash
cd ${ISAAC_ROS_WS}/src/isaac_ros_common && ./scripts/run_dev.sh
#Run the rosbag file to simulate an image stream:
admin@lkk-intel12:/workspaces/isaac_ros-dev$ ros2 bag play -l ${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_apriltag/quickstart.bag --remap image:=image_rect camera_info:=camera_info_rect
```

To visualize the results, Open a new terminal inside the Docker container:
```bash
cd ${ISAAC_ROS_WS}/src/isaac_ros_common && \
   ./scripts/run_dev.sh
admin@lkk-intel12:/workspaces/isaac_ros-dev$ ros2 topic list
/camera_info_rect
/camera_info_rect/nitros
/events/read_split
/image_rect
/image_rect/nitros
/parameter_events
/rosout
/tag_detections
/tag_detections/nitros
admin@lkk-intel12:/workspaces/isaac_ros-dev$ ros2 topic echo /tag_detections
```

Install Camera
```bash
cd ${ISAAC_ROS_WS}/src && git clone -b ros2 https://github.com/ros-drivers/usb_cam
cd /workspaces/isaac_ros-dev && \
  colcon build --symlink-install && \
  source install/setup.bash
ros2 launch isaac_ros_apriltag isaac_ros_apriltag_usb_cam.launch.py camera_width:=1280 camera_height:=720
```

Open another terminal, launch rviz2
```
rviz2 -d $(ros2 pkg prefix isaac_ros_apriltag --share)/rviz/usb_cam.rviz
```

## Setup Realsense
https://www.intelrealsense.com/get-started-depth-camera/
Option1: Realsense DKMS kernel drivers package (librealsense2-dkms)
https://github.com/IntelRealSense/librealsense/blob/development/doc/distribution_linux.md

Option2: Configuring and building from the source code
https://dev.intelrealsense.com/docs/compiling-librealsense-for-linux-ubuntu-guide
https://github.com/IntelRealSense/librealsense/blob/development/doc/installation.md
```bash
sudo apt-get update && sudo apt-get upgrade
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
sudo apt-get install apt-transport-https
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt-get update
sudo apt-get install librealsense2-dkms
sudo apt-get install librealsense2-utils
$ lsusb #plugin the realsense camera
Bus 004 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 003 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
Bus 002 Device 002: ID 8086:0b3a Intel Corp. Intel(R) RealSense(TM) Depth Camera 435i
realsense-viewer

(mycondapy310) lkk@lkk-intel12:~/miniconda3/lib$ realsense-viewer
realsense-viewer: /home/lkk/miniconda3/lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/librealsense2.so.2.55)
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
$ export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH #solved the problem
```

Configure the container created by isaac_ros_common/scripts/run_dev.sh to include librealsense and realsense2-camera. Create the .isaac_ros_common-config file in the isaac_ros_common/scripts directory:
```bash
isaac_ros-dev/src$ git clone https://github.com/IntelRealSense/realsense-ros.git -b 4.51.1
cd ${ISAAC_ROS_WS}/src/isaac_ros_common/scripts && \
touch .isaac_ros_common-config && \
echo CONFIG_IMAGE_KEY=ros2_humble.realsense > .isaac_ros_common-config
#Launch the Docker container:
$ cd ${ISAAC_ROS_WS}/src/isaac_ros_common && \
./scripts/run_dev.sh -d ${ISAAC_ROS_WS}
#Build the realsense_splitter and realsense2* packages:
cd ${ISAAC_ROS_WS} && \
colcon build --symlink-install --packages-up-to-regex realsense*
#After the container image is rebuilt and you are inside the container, you can run realsense-viewer to verify that the RealSense camera is connected.
realsense-viewer
```

https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_pipeline/isaac_ros_image_proc/index.html#quickstart
```bash
cd ${ISAAC_ROS_WS}/src && \
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline.git
rosdep install --from-paths ${ISAAC_ROS_WS}/src/isaac_ros_image_pipeline/isaac_ros_image_proc --ignore-src -y
cd ${ISAAC_ROS_WS}/ && \
   colcon build --symlink-install --packages-up-to isaac_ros_image_proc --symlink-install
source install/setup.bash
sudo apt-get install -y ros-humble-isaac-ros-examples ros-humble-isaac-ros-realsense

ros2 launch isaac_ros_examples isaac_ros_examples.launch.py launch_fragments:=realsense_mono,resize
```
In another terminal
```bash
ros2 run image_view image_view --ros-args --remap image:=resize/image
```
## Docker changes
Commit docker images changes:
```bash
(mycondapy310) lkk@lkk-intel12:~/miniconda3/lib$ docker ps
CONTAINER ID   IMAGE                  COMMAND                  CREATED        STATUS        PORTS     NAMES
d0b8a15dfacd   isaac_ros_dev-x86_64   "/usr/local/bin/scri…"   14 hours ago   Up 14 hours             isaac_ros_dev-x86_64-container
(mycondapy310) lkk@lkk-intel12:~/miniconda3/lib$ docker commit -a "Kaikai Liu" -m "First ROS2-x86 container" d0b8a15dfacd myisaac_ros:v1
(mycondapy310) lkk@lkk-intel12:~/miniconda3/lib$ docker images
REPOSITORY                                      TAG                        IMAGE ID       CREATED          SIZE
myisaac_ros                                     v1                         6ffb5c654ba5   46 seconds ago   30.8GB
isaac_ros_dev-x86_64                            latest                     8b15e5a14953   15 hours ago     30.6GB
```

## ISAAC SIM
https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html
https://docs.omniverse.nvidia.com/isaacsim/latest/introductory_tutorials/tutorial_intro_workflows.html#ros-application
https://docs.omniverse.nvidia.com/isaacsim/latest/ros2_tutorials/index.html

## References
https://developer.nvidia.com/isaac/ros
https://nvidia-isaac-ros.github.io/getting_started/index.html

https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_apriltag/isaac_ros_apriltag/index.html#quickstart
Tutorial for AprilTag Detection with a USB Camera: https://nvidia-isaac-ros.github.io/concepts/fiducials/apriltag/tutorial_usb_cam.html

https://nvidia-isaac-ros.github.io/getting_started/hardware_setup/sensors/index.html
https://nvidia-isaac-ros.github.io/getting_started/hardware_setup/sensors/realsense_setup.html


ISAAC ROS All repositories: https://nvidia-isaac-ros.github.io/repositories_and_packages/index.html

https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_common/index.html


https://developer.nvidia.com/isaac/sim
https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html

https://developer.nvidia.com/blog/simulate-and-localize-a-husky-robot-with-nvidia-isaac/

# Robotic Platforms
https://nvidia-isaac-ros.github.io/robots/nova_developer_kit/index.html#nova-developer-kit
https://nvidia-isaac-ros.github.io/robots/nova_carter/index.html#nova-carter