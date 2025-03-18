#!/bin/sh

# Name of the Docker image
IMAGE_NAME="text-to-sql-app"

# Build the Docker image from the Dockerfile in the current directory.
echo "Building Docker image '${IMAGE_NAME}'..."
docker build -t ${IMAGE_NAME} .

# Check if the build succeeded.
if [ $? -ne 0 ]; then
    echo "Docker build failed. Please check the Dockerfile and try again."
    exit 1
fi

# Run the Docker container interactively.
echo "Running the Docker container..."
docker run -it ${IMAGE_NAME}
