#!/bin/bash

# SimplerEnv Docker Image Build Script
# Usage: bash docker/build_docker.sh

set -e

IMAGE_NAME="simplerenv"
TAG="benchmark-v2"

echo "Building Docker image: ${IMAGE_NAME}:${TAG}"
echo "This may take a while (30-60 minutes)..."

docker build \
  -f docker/Dockerfile \
  -t ${IMAGE_NAME}:${TAG} \
  .
