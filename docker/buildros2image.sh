#!/bin/bash
set -e
BASE_IMAGE=ubuntu22cu117:latest
CUDA_MAJOR=11 			# CUDA major version
CUDA_MINOR=7 			# CUDA minor version 


TAG="myros2humblecuda${CUDA_MAJOR}"
DOCKERFILE="Dockerfile.ros2humble"

echo "Building '${TAG}'" 

docker build \
--build-arg BASE_IMAGE=${BASE_IMAGE} \
--build-arg CUDA_MAJOR=${CUDA_MAJOR} \
--build-arg CUDA_MINOR=${CUDA_MINOR} \
-t "${TAG}" -f "${DOCKERFILE}" .

#Successfully tagged mycuda:latest
#xhost +si:localuser:root  # allows container to communicate with X server
#docker run  --gpus all --runtime nvidia --privileged -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix <container_tag> # run the docker container
#docker run  --gpus all --runtime nvidia --privileged -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix mycuda:latest /bin/bash