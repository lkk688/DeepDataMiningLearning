#!/bin/bash
set -e
BASE_IMAGE=nvidia/cuda:11.7.1-devel-ubuntu22.04
CUDA_MAJOR=11 			# CUDA major version
CUDA_MINOR=7 			# CUDA minor version 


TAG="mycuda${CUDA_MAJOR}"
DOCKERFILE="Dockerfile.ubuntu22cu117"

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