#!/bin/sh

IMAGE_NAME="text-to-sql-app"

echo "Building Docker image '${IMAGE_NAME}'..."
docker build -t ${IMAGE_NAME} .

if [ $? -ne 0 ]; then
    echo "Docker build failed. Please check the Dockerfile and try again."
    exit 1
fi

echo "Running the Docker container..."
docker run -it ${IMAGE_NAME}
