#!/bin/bash

# SimplerEnv Docker Container Run Script
# Usage: bash docker/run_docker.sh

set -e

IMAGE_NAME="simplerenv"
TAG="benchmark-v2"
CONTAINER_NAME="simplerenv"

# Dockerコンテナを起動し、現在のユーザーのDISPLAY環境変数とX11のソケットを渡す
docker run --gpus all -it -d \
  --name ${CONTAINER_NAME} \
  --shm-size=250g \
  -e DISPLAY=$DISPLAY \
  -v $(pwd):/workspace \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/.Xauthority:/root/.Xauthority \
  --net host \
  ${IMAGE_NAME}:${TAG}

echo ""
echo "Container started successfully!"
echo "Container name: ${CONTAINER_NAME}"
echo ""
echo "To enter the container, run:"
echo "  docker exec -it ${CONTAINER_NAME} bash"
echo ""
echo "To stop the container, run:"
echo "  docker stop ${CONTAINER_NAME}"

